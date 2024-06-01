import functools
import io
import os
import urllib.request

# Disable future warnings.
import warnings

import ase
import ase.calculators.calculator as ase_calc
import ase.io as ase_io
import ase.optimize as ase_opt
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet

import e3x

# import py3Dmol

warnings.simplefilter(action="ignore", category=FutureWarning)


def prepare_datasets(key, num_train, num_valid):
    # Load the dataset.
    dataset = np.load(filename)

    # Make sure that the dataset contains enough entries.
    num_data = len(dataset["E"])
    num_draw = num_train + num_valid
    if num_draw > num_data:
        raise RuntimeError(
            f"datasets only contains {num_data} points, requested num_train={num_train}, num_valid={num_valid}"
        )

    # Randomly draw train and validation sets from dataset.
    choice = np.asarray(
        jax.random.choice(key, num_data, shape=(num_draw,), replace=False)
    )
    train_choice = choice[:num_train]
    valid_choice = choice[num_train:]

    # Determine mean energy of the training set.
    mean_energy = np.mean(dataset["E"][train_choice])  # ~ -97000

    # Collect and return train and validation sets.
    train_data = dict(
        energy=jnp.asarray(dataset["E"][train_choice, 0] - mean_energy),
        forces=jnp.asarray(dataset["F"][train_choice]),
        atomic_numbers=jnp.asarray(dataset["z"]),
        positions=jnp.asarray(dataset["R"][train_choice]),
    )
    valid_data = dict(
        energy=jnp.asarray(dataset["E"][valid_choice, 0] - mean_energy),
        forces=jnp.asarray(dataset["F"][valid_choice]),
        atomic_numbers=jnp.asarray(dataset["z"]),
        positions=jnp.asarray(dataset["R"][valid_choice]),
    )
    return train_data, valid_data, mean_energy


class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 118  # This is overkill for most applications.

    def energy(
        self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
    ):
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )

        # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1, features=self.features
        )(atomic_numbers)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
                # features for efficiency reasons.
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )
                # After the final message pass, we can safely throw away all non-scalar features.
                x = e3x.nn.change_max_degree_or_type(
                    x, max_degree=0, include_pseudotensors=False
                )
            else:
                # In intermediate iterations, the message-pass should consider all possible coupling paths.
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            y = e3x.nn.add(x, y)

            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

            # Residual connection.
            x = e3x.nn.add(x, y)

        # 5. Predict atomic energies with an ordinary dense layer.
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros
        )(
            x
        )  # (..., Natoms, 1, 1, 1)
        atomic_energies = jnp.squeeze(
            atomic_energies, axis=(-1, -2, -3)
        )  # Squeeze last 3 dimensions.
        atomic_energies += element_bias[atomic_numbers]

        # 6. Sum atomic energies to obtain the total energy.
        energy = jax.ops.segment_sum(
            atomic_energies, segment_ids=batch_segments, num_segments=batch_size
        )

        # To be able to efficiently compute forces, our model should return a single output (instead of one for each
        # molecule in the batch). Fortunately, since all atomic contributions only influence the energy in their own
        # batch segment, we can simply sum the energy of all molecules in the batch to obtain a single proxy output
        # to differentiate.
        return (
            -jnp.sum(energy),
            energy,
        )  # Forces are the negative gradient, hence the minus sign.

    @nn.compact
    def __call__(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments=None,
        batch_size=None,
    ):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        # Since we want to also predict forces, i.e. the gradient of the energy w.r.t. positions (argument 1), we use
        # jax.value_and_grad to create a function for predicting both energy and forces for us.
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
        (_, energy), forces = energy_and_forces(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
        )

        return energy, forces


def prepare_batches(key, data, batch_size):
    # Determine the number of training steps per epoch.
    data_size = len(data["energy"])
    steps_per_epoch = data_size // batch_size

    # Draw random permutations for fetching batches from the train data.
    perms = jax.random.permutation(key, data_size)
    perms = perms[
        : steps_per_epoch * batch_size
    ]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.
    num_atoms = len(data["atomic_numbers"])
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    atomic_numbers = jnp.tile(data["atomic_numbers"], batch_size)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)

    # Assemble and return batches.
    return [
        dict(
            energy=data["energy"][perm],
            forces=data["forces"][perm].reshape(-1, 3),
            atomic_numbers=atomic_numbers,
            positions=data["positions"][perm].reshape(-1, 3),
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
        )
        for perm in perms
    ]


@jax.jit
def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
    return message_passing_model.apply(
        params,
        atomic_numbers=atomic_numbers,
        positions=positions,
        dst_idx=dst_idx,
        src_idx=src_idx,
    )


class MessagePassingCalculator(ase_calc.Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self, atoms, properties, system_changes=ase.calculators.calculator.all_changes
    ):
        print("pase")
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)
        print("pase")
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atoms))
        energy, forces = evaluate_energies_and_forces(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            dst_idx=dst_idx,
            src_idx=src_idx,
        )

        print("pase", ase.units.kcal, ase.units.mol, energy)
        self.results["energy"] = energy
        print("pase", ase.units.kcal, ase.units.mol, forces)
        self.results["forces"] = np.array(forces[0])
        return self.results


filename = "/home/beemoqc2/Documents/e3x/docs/source/examples/datos_modificados.npz"

# Model hyperparameters.
features = 32
max_degree = 1
num_iterations = 5
num_basis_functions = 16
cutoff = 10.0
max_atomic_number = 26

# Training hyperparameters.
num_train = 4000
num_valid = 1
num_epochs = 1
learning_rate = 0.001
forces_weight = 0.9
batch_size = 10
import pickle

with open(
    "/home/beemoqc2/Documents/e3x/docs/source/examples/model_params_best_model.pkl",
    "rb",
) as file:
    params = pickle.load(file)

# Create PRNGKeys.
data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

# Draw training and validation sets.
train_data, valid_data, _ = prepare_datasets(
    data_key, num_train=num_train, num_valid=num_valid
)

# Create and train model.
message_passing_model = MessagePassingModel(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
)

# Initialize atoms object and attach calculator.
atoms = ase.Atoms(valid_data["atomic_numbers"], valid_data["positions"][0])
atoms.set_calculator(MessagePassingCalculator())

# Run structure optimization with BFGS.
a = ase_opt.BFGS(atoms).run(fmax=0.05)
