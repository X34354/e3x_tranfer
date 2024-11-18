import functools
import os
import pickle

# Disable future warnings.
import warnings
from datetime import datetime
from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
import plotly.graph_objs as go
import plotly.io as pio
from jax import device_put
from plotly.subplots import make_subplots

import e3x

warnings.simplefilter(action="ignore", category=FutureWarning)
jax.devices()


def create_loss_plot(
    train_loss: List[np.ndarray],
    val_loss: List[np.ndarray],
    train_label: str,
    val_label: str,
    title: str,
    filename: str,
) -> None:
    """
    Create a Plotly figure with training and validation loss curves and save it as an HTML file.

    Args:
        train_loss (List[np.ndarray]): List of training loss values.
        val_loss (List[np.ndarray]): List of validation loss values.
        train_label (str): Label for the training loss curve.
        val_label (str): Label for the validation loss curve.
        title (str): Title of the plot.
        filename (str): Filename to save the HTML file.

    Returns:
        None
    """
    train_loss_list = [float(loss) for loss in train_loss]
    val_loss_list = [float(loss) for loss in val_loss]

    trace_train = go.Scatter(y=train_loss_list, mode="lines", name=train_label)
    trace_val = go.Scatter(y=val_loss_list, mode="lines", name=val_label)

    fig = go.Figure()
    fig.add_trace(trace_train)
    fig.add_trace(trace_val)
    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title="Loss", legend_title="Legend"
    )
    pio.write_html(fig, filename)
    mlflow.log_artifact(filename)


def prepare_datasets(key, num_train, num_valid, filename):
    # Load the dataset.
    dataset = np.load(filename)

    if rm_atom_energ:
        dataset["E"][:] -= atom_energ
        print("--> Atomic energies removed")

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
    print(f"mean_energy: {mean_energy}")

    # Collect and return train and validation sets.
    train_data = dict(
        energy=jnp.asarray(dataset["E"][train_choice, 0] - mean_energy),
        forces=jnp.asarray(dataset["F"][train_choice]),
        atomic_numbers=jnp.asarray(dataset["z"]),
        # atomic_numbers=jnp.asarray(z_hack),
        positions=jnp.asarray(dataset["R"][train_choice]),
    )
    valid_data = dict(
        energy=jnp.asarray(dataset["E"][valid_choice, 0] - mean_energy),
        forces=jnp.asarray(dataset["F"][valid_choice]),
        atomic_numbers=jnp.asarray(dataset["z"]),
        # atomic_numbers=jnp.asarray(z_hack),
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


def mean_squared_loss(
    energy_prediction, energy_target, forces_prediction, forces_target, forces_weight
):
    energy_loss = jnp.mean(optax.l2_loss(energy_prediction, energy_target))
    forces_loss = jnp.mean(optax.l2_loss(forces_prediction, forces_target))
    return energy_loss + forces_weight * forces_loss


def mean_absolute_error(prediction, target):
    return jnp.mean(jnp.abs(prediction - target))


@functools.partial(
    jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size")
)
def train_step(
    model_apply, optimizer_update, batch, batch_size, forces_weight, opt_state, params
):
    def loss_fn(params):
        energy, forces = model_apply(
            params,
            atomic_numbers=batch["atomic_numbers"],
            positions=batch["positions"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )
        loss = mean_squared_loss(
            energy_prediction=energy,
            energy_target=batch["energy"],
            forces_prediction=forces,
            forces_target=batch["forces"],
            forces_weight=forces_weight,
        )
        return loss, (energy, forces)

    (loss, (energy, forces)), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)

    updates, opt_state = optimizer_update(grad, opt_state, params)

    params = optax.apply_updates(params, updates)

    energy_mae = mean_absolute_error(energy, batch["energy"])
    forces_mae = mean_absolute_error(forces, batch["forces"])

    return params, opt_state, loss, energy_mae, forces_mae


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size"))
def eval_step(model_apply, batch, batch_size, forces_weight, params):
    energy, forces = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    loss = mean_squared_loss(
        energy_prediction=energy,
        energy_target=batch["energy"],
        forces_prediction=forces,
        forces_target=batch["forces"],
        forces_weight=forces_weight,
    )
    energy_mae = mean_absolute_error(energy, batch["energy"])
    forces_mae = mean_absolute_error(forces, batch["forces"])
    return loss, energy_mae, forces_mae


def train_model(
    key,
    model,
    train_data,
    valid_data,
    num_epochs,
    learning_rate,
    forces_weight,
    batch_size,
    file,
):
    # Initialize model parameters and optimizer state.
    print("Initializing model parameters and optimizer state.")
    key, init_key = jax.random.split(key)

    if str_optim == "adam":
        optimizer = optax.adam(learning_rate)
    elif str_optim == "adabelief":
        optimizer = optax.adabelief(learning_rate)
    elif str_optim == "lamb":
        optimizer = optax.lamb(learning_rate)
    elif str_optim == "lion":
        optimizer = optax.lion(0.1 * learning_rate, weight_decay=0.005)

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(
        len(train_data["atomic_numbers"])
    )
    with open(
        model_train_pickle,
        "rb",
    ) as file:
        loaded_params = pickle.load(file)

    params = loaded_params

    opt_state = optimizer.init(params)

    # Batches for the validation set need to be prepared only once.
    print("Preparing batches for the validation set.")
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    # Train for 'num_epochs' epochs.
    print("Starting training loop...")
    loss_train = []
    loss_energy = []
    loss_force = []
    loss_energy_rmse = []
    loss_force_rmse = []

    loss_val_train = []
    loss_val_energy = []
    loss_val_force = []
    loss_val_energy_rmse = []
    loss_val_force_rmse = []

    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)

        # Loop over train batches.
        train_loss = 0.0
        train_energy_mae = 0.0
        train_forces_mae = 0.0
        train_energy_rmse = 0.0
        train_forces_rmse = 0.0

        for i, batch in enumerate(train_batches):
            (
                params,
                opt_state,
                loss,
                energy_mae,
                forces_mae,
                energy_rmse,
                forces_rmse,
            ) = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                forces_weight=forces_weight,
                opt_state=opt_state,
                params=params,
            )

            train_loss += (loss - train_loss) / (i + 1)
            train_energy_mae += (energy_mae - train_energy_mae) / (i + 1)
            train_forces_mae += (forces_mae - train_forces_mae) / (i + 1)
            train_energy_rmse += (energy_rmse - train_energy_rmse) / (i + 1)
            train_forces_rmse += (forces_rmse - train_forces_rmse) / (i + 1)

        loss_train.append(train_loss)
        loss_energy.append(train_energy_mae)
        loss_force.append(train_forces_mae)
        loss_energy_rmse.append(train_energy_rmse)
        loss_force_rmse.append(train_forces_rmse)

        # Evaluate on validation set.
        valid_loss = 0.0
        valid_energy_mae = 0.0
        valid_forces_mae = 0.0
        valid_energy_rmse = 0.0
        valid_forces_rmse = 0.0

        for i, batch in enumerate(valid_batches):
            loss, energy_mae, forces_mae, energy_rmse, forces_rmse = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                forces_weight=forces_weight,
                params=params,
            )

            valid_loss += (loss - valid_loss) / (i + 1)
            valid_energy_mae += (energy_mae - valid_energy_mae) / (i + 1)
            valid_forces_mae += (forces_mae - valid_forces_mae) / (i + 1)
            valid_energy_rmse += (energy_rmse - valid_energy_rmse) / (i + 1)
            valid_forces_rmse += (forces_rmse - valid_forces_rmse) / (i + 1)

        loss_val_train.append(valid_loss)
        loss_val_energy.append(valid_energy_mae)
        loss_val_force.append(valid_forces_mae)
        loss_val_energy_rmse.append(valid_energy_rmse)
        loss_val_force_rmse.append(valid_forces_rmse)

        # Print progress.
        print(f"Epoch: {epoch: 3d}                    Train:      Valid:")
        print(f"    Loss [a.u.]             {train_loss : 8.3f} {valid_loss : 8.3f}")
        print(
            f"    Energy MAE [kcal/mol]   {train_energy_mae: 8.3f} {valid_energy_mae: 8.3f}"
        )
        print(
            f"    Energy RMSE [kcal/mol]  {train_energy_rmse: 8.3f} {valid_energy_rmse: 8.3f}"
        )
        print(
            f"    Forces MAE [kcal/mol/Å] {train_forces_mae: 8.3f} {valid_forces_mae: 8.3f}"
        )
        print(
            f"    Forces RMSE [kcal/mol/Å]{train_forces_rmse: 8.3f} {valid_forces_rmse: 8.3f}"
        )

        # Best model tracking
        if epoch == 1:
            valid_loss_best = 1e10
            valid_energy_mae_best = 1e10
            valid_forces_mae_best = 1e10
            valid_energy_rmse_best = 1e10
            valid_forces_rmse_best = 1e10

        if jnp.isnan(train_forces_mae) or jnp.isnan(valid_forces_mae):
            print("BREAK: NaN appeared")
            print(f"--------- BEST MODEL --------")
            print(f"    Loss [a.u.]             {valid_loss_best : 8.3f}")
            print(f"    Energy MAE [kcal/mol]   {valid_energy_mae_best: 8.3f}")
            print(f"    Energy RMSE [kcal/mol]  {valid_energy_rmse_best: 8.3f}")
            print(f"    Forces MAE [kcal/mol/Å] {valid_forces_mae_best: 8.3f}")
            print(f"    Forces RMSE [kcal/mol/Å]{valid_forces_rmse_best: 8.3f}")
            break

        if valid_forces_mae < valid_forces_mae_best:
            print("Updating best model...")
            params_best = params.copy()
            valid_loss_best = valid_loss
            valid_energy_mae_best = valid_energy_mae
            valid_forces_mae_best = valid_forces_mae
            valid_energy_rmse_best = valid_energy_rmse
            valid_forces_rmse_best = valid_forces_rmse
            train_energy_mae_best = train_energy_mae
            train_forces_mae_best = train_forces_mae
            # Store best model
            params_best = device_put(params_best)
            params_2save = {
                "errors": [
                    valid_loss_best,
                    valid_energy_mae_best,
                    valid_forces_mae_best,
                ],
                "model": params_best,
            }
            # Save model if needed
            # jnp.savez(f"best_model_{tail_str}_tmp_test.npz", **params_2save)

        if os.path.exists("early_stop"):
            print("BREAK: early_stop")
            print(f"--------- BEST MODEL --------")
            print(f"    Loss [a.u.]             {valid_loss_best : 8.3f}")
            print(f"    Energy MAE [kcal/mol]   {valid_energy_mae_best: 8.3f}")
            print(f"    Energy RMSE [kcal/mol]  {valid_energy_rmse_best: 8.3f}")
            print(f"    Forces MAE [kcal/mol/Å] {valid_forces_mae_best: 8.3f}")
            print(f"    Forces RMSE [kcal/mol/Å]{valid_forces_rmse_best: 8.3f}")
            break

    # Return final model parameters.
    print(f"--------- BEST MODEL --------")
    print(f"    Loss [a.u.]             {valid_loss_best : 8.3f}")
    print(f"    Energy MAE [kcal/mol]   {valid_energy_mae_best: 8.3f}")
    print(f"    Energy RMSE [kcal/mol]  {valid_energy_rmse_best: 8.3f}")
    print(f"    Forces MAE [kcal/mol/Å] {valid_forces_mae_best: 8.3f}")
    print(f"    Forces RMSE [kcal/mol/Å]{valid_forces_rmse_best: 8.3f}")
    return (
        params_best,
        loss_val_train,
        loss_val_energy,
        loss_val_force,
        loss_val_energy_rmse,
        loss_val_force_rmse,
        loss_train,
        loss_energy,
        loss_force,
        loss_energy_rmse,
        loss_force_rmse,
        train_energy_mae_best,
        train_forces_mae_best,
        valid_loss_best,
        valid_energy_mae_best,
        valid_forces_mae_best,
        valid_energy_rmse_best,
        valid_forces_rmse_best,
    )


name_data_retrain = "Si16Vplus..DFT.SP-GRD.B3LYP.tight.Data.5628.R_E_F_D_Q"
path_data = "../e3x_tranfer/data/"
path_model = "../e3x_tranfer/model/"
experiment_name = "MP_Model_Training_fine_tunig"

model_train_pickle = f"{path_model}best_model_1phase.pkl"
data_train = None
data_retrain = f"{path_data}{name_data_retrain}_retrain.npz"

model_retrain_pickle_save = f"{path_model}{name_data_retrain}_retrain_.pkl"

# Atomic energies: V + 16 * Si
rm_atom_energ = True
atom_energ = -3507661.3898417074  # kcal/mol
# --- Model hyperparameters ---
features = 32
max_degree = 1
num_iterations = 5
num_basis_functions = 16
cutoff = 10.0
max_atomic_number = 26
str_optim = "adam"
# -----------------------------


# ---- Training hyperparameters ----
num_train = 4000
num_valid = 1000
num_epochs = 2000
learning_rate = 0.001
forces_weight = 0.9
batch_size = 256
# ----------------------------------

import os

import yaml

# Cargar el archivo YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Acceder a las variables de paths
paths = config["paths"]
name_data_retrain = paths["name_data_retrain"]
path_data = paths["path_data"]
path_model = paths["path_model"]
experiment_name = paths["experiment_name"]

model_train_pickle = f"{path_model}best_model_1phase.pkl"

data_train = paths["data_train"]  # Será None

data_retrain = f"{path_data}{name_data_retrain}.npz"
model_retrain_pickle_save = f"{path_model}{name_data_retrain}.pkl"

# Acceder a las variables de energías atómicas
atom_energies = config["atom_energies"]
rm_atom_energ = atom_energies["rm_atom_energ"]
atom_energ = atom_energies["atom_energ"]

# Acceder a los hiperparámetros del modelo
model_hyperparams = config["model_hyperparameters"]
features = model_hyperparams["features"]
max_degree = model_hyperparams["max_degree"]
num_iterations = model_hyperparams["num_iterations"]
num_basis_functions = model_hyperparams["num_basis_functions"]
cutoff = model_hyperparams["cutoff"]
max_atomic_number = model_hyperparams["max_atomic_number"]
str_optim = model_hyperparams["str_optim"]

# Acceder a los hiperparámetros de entrenamiento
training_hyperparams = config["training_hyperparameters"]
num_train = training_hyperparams["num_train"]
num_valid = training_hyperparams["num_valid"]
num_epochs = training_hyperparams["num_epochs"]
learning_rate = training_hyperparams["learning_rate"]
forces_weight = training_hyperparams["forces_weight"]
batch_size = training_hyperparams["batch_size"]

# Ejemplo de uso de las variables

if __name__ == "__main__":

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=f"run_{name_data_retrain}_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        mlflow.log_artifact("config.yaml")
        mlflow.log_artifact(data_retrain)
        mlflow.log_artifact(model_train_pickle)
        # Create PRNGKeys.
        data_key, train_key = jax.random.split(jax.random.PRNGKey(0), 2)

        # Draw training and validation sets.
        train_data, valid_data, _ = prepare_datasets(
            data_key, num_train=num_train, num_valid=num_valid, filename=data_retrain
        )

        # Create and train model.
        print("Creating MP model...")
        message_passing_model = MessagePassingModel(
            features=features,
            max_degree=max_degree,
            num_iterations=num_iterations,
            num_basis_functions=num_basis_functions,
            cutoff=cutoff,
            max_atomic_number=max_atomic_number,
        )
        print("Creating MP model [DONE]")

        print("Start training model...")
        (
            params_best,
            loss_val_train,
            loss_val_energy,
            loss_val_force,
            loss_val_energy_rmse,
            loss_val_force_rmse,
            loss_train,
            loss_energy,
            loss_force,
            loss_energy_rmse,
            loss_force_rmse,
            train_energy_mae_best,
            train_forces_mae_best,
            valid_loss_best,
            valid_energy_mae_best,
            valid_forces_mae_best,
            valid_energy_rmse_best,
            valid_forces_rmse_best,
        ) = train_model(
            key=train_key,
            model=message_passing_model,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            forces_weight=forces_weight,
            batch_size=batch_size,
            file=model_train_pickle,
        )

        # Log metrics
        mlflow.log_metric("final_train_loss", loss_train[-1])
        mlflow.log_metric("final_val_loss", loss_val_train[-1])
        mlflow.log_metric("final_energy_mae", loss_energy[-1])
        mlflow.log_metric("final_force_mae", loss_force[-1])
        mlflow.log_metric("train_best_force_mae", train_forces_mae_best)
        mlflow.log_metric("train_best_enery_mae", train_energy_mae_best)
        mlflow.log_metric("val_best_force_mae", valid_forces_mae_best)
        mlflow.log_metric("val_best_enery_mae", valid_energy_mae_best)
        mlflow.log_metric("best_loss", valid_loss_best)

        with open(model_retrain_pickle_save, "wb") as f:
            pickle.dump(params, f)

        mlflow.log_artifact(model_retrain_pickle_save)
        # Create and log plots
        create_loss_plot(
            loss_train,
            loss_val_train,
            "Training Loss",
            "Validation Loss",
            "Training vs Validation Loss (Train)",
            "train_vs_val_train.html",
        )
        create_loss_plot(
            loss_energy,
            loss_val_energy,
            "Training Loss Energy",
            "Validation Loss Energy",
            "Training vs Validation Loss (Energy)",
            "train_vs_val_energy.html",
        )
        create_loss_plot(
            loss_force,
            loss_val_force,
            "Training Loss Force",
            "Validation Loss Force",
            "Training vs Validation Loss (Force)",
            "train_vs_val_force.html",
        )

        print("Experiment completed and logged with MLflow.")
        mlflow.end_run()
