import os

# Establecer la GPU visible en el entorno del notebook (GPU 1 es la RTX 3090)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import functools
import e3x
from flax import linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import functools
import jax
import jax.numpy as jnp
from flax import linen as nn
import e3x
# Disable future warnings.
import warnings
import pickle
warnings.simplefilter(action="ignore", category=FutureWarning)
jax.devices()

from math import acos, degrees
import numpy as np
from numpy import linalg as la
from sklearn.metrics import mean_absolute_error, root_mean_squared_error



class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 118  # Esto es excesivo para la mayoría de las aplicaciones.

    def energy(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
        # 1. Calcular vectores de desplazamiento.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Forma (num_pairs, 3).

        # 2. Expandir los vectores de desplazamiento en funciones base.
        basis = e3x.nn.basis(  # Forma (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )

        # 3. Embedding de números atómicos en el espacio de características, x tiene forma (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number+1, features=self.features)(atomic_numbers)

        # 4. Realizar iteraciones (message-passing + refinamiento atómico).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Iteración final.
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(x, basis, dst_idx=dst_idx, src_idx=src_idx)
                x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
            else:
                y = e3x.nn.MessagePass()(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            y = e3x.nn.add(x, y)

            # Refinamiento atómico MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

            # Conexión residual.
            x = e3x.nn.add(x, y)

        # 5. Predecir cargas parciales atómicas con una capa densa.
        charges = e3x.nn.Dense(1, use_bias=True)(x)  # (..., num_atoms, 1, 1, 1)
        charges = jnp.squeeze(charges, axis=(-1, -2, -3))  # Eliminar últimas 3 dimensiones.

        # 6. Calcular el momento dipolar.
        # Multiplicar cargas parciales por posiciones y segmentar por batch.
        dipole_contributions = charges[:, None] * positions  # (num_atoms, 3)
        dipole_moment = jax.ops.segment_sum(dipole_contributions, segment_ids=batch_segments, num_segments=batch_size)

        # Retornar el momento dipolar total sumando sobre las moléculas en el batch.
        #total_dipole_moment = jnp.sum(dipole_moment, axis=0)  # Vector de dimensión (3,)

        # No necesitamos calcular fuerzas, así que podemos retornar directamente el momento dipolar.
        return dipole_moment

    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        # Llamamos al método energy para calcular el momento dipolar.
        dipole_moment = self.energy(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)

        return dipole_moment

def angle_between(a, b):

    theta_degrees = degrees(acos((np.dot(a, b)) / (la.norm(a) * la.norm(b))))
    return theta_degrees

def prepare_trayect(filename):
    # Cargar el dataset completo
    dataset = np.load(filename)

    Z = jnp.full(1, 23)
    Z = jnp.append(Z, jnp.full(16, 14))
    Z = jnp.expand_dims(Z, axis=0)
    #Z = jnp.repeat(Z, num_data, axis=0)




    # Collect and return train and validation sets.
    train_data = dict(
        #atomic_numbers=jnp.asarray(Z[train_choice]),
        atomic_numbers=jnp.asarray(Z),
        # atomic_numbers=jnp.asarray(z_hack),
        positions=jnp.asarray(dataset["R"]),
    )


    return train_data

def prepare_batches_no_shuffle(data, batch_size):
    # Determine the number of batches.
    data_size = len(trayect["positions"])
    steps_per_epoch = data_size // batch_size

    # Get indices for the batches without shuffling.
    indices = np.arange(data_size)
    indices = indices[:steps_per_epoch * batch_size]
    indices = indices.reshape((steps_per_epoch, batch_size))

    # Prepare entries that are identical for each batch.
    num_atoms = len(data["atomic_numbers"][0])
    batch_segments = jnp.repeat(jnp.arange(batch_size), num_atoms)
    atomic_numbers = jnp.tile(data["atomic_numbers"][0], batch_size)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)

    # Precompute batch constants for caching.
    batch_constants = {
        'atomic_numbers': atomic_numbers,
        'dst_idx': dst_idx,
        'src_idx': src_idx,
        'batch_segments': batch_segments,
    }

    # Assemble and return batches.
    batches = []
    for idx in indices:
        batch = {
            'atomic_numbers': batch_constants['atomic_numbers'],
            'positions': data['positions'][idx].reshape(-1, 3),
            'dst_idx': batch_constants['dst_idx'],
            'src_idx': batch_constants['src_idx'],
            'batch_segments': batch_constants['batch_segments'],
        }
        batches.append(batch)

    return batches

num_train = 4900
num_val = 1000
# Define training hyperparameters.
learning_rate = 0.001
num_epochs = 2000
batch_size = 512
# Model hyperparameters.
features = 128
max_degree = 2
num_iterations = 3
num_basis_functions = 16
cutoff = 6.0
max_atomic_number = 23
batch_size = 512
model_save_path = "mode_training_Si16Vplus..DFT.SP-GRD.wB97X-D.tight.Data.5986.R_E_F_D_Q.pkl"

filename_trayec = "../e3x_tranfer/SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_400K_1B_01_POSITION_0.npz"
filename_trayec_2_reshape = "../e3x_tranfer/SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_400K_1B_01_POSITION_0_reshape.npz"

if __name__ == "__main__":


    with open(model_save_path, "rb") as file:
        params = pickle.load(file)

    model = MessagePassingModel(
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
        max_atomic_number=max_atomic_number
    )
    
    dataset = np.load(filename_trayec, allow_pickle=True)
    if len(dataset['R'].shape) == 4 : 
        print('dataset 4 shapes')
        dataset_modified = {
            key: np.squeeze(value, axis=0) if key == "R" else value
            for key, value in dataset.items()
        }

        # Guardar el dataset modificado
        np.savez(
            filename_trayec_2_reshape,
            **dataset_modified,
            allow_pickle=False,
        )
        filename_trayec = filename_trayec_2_reshape

    trayect= prepare_trayect(
        filename_trayec
    )

    trayect['positions']  = trayect['positions']  - trayect['positions'] [:, 0:1, :]
    data_final = []
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(trayect["atomic_numbers"][0]))

    i = 0
    batches  = prepare_batches_no_shuffle(trayect, batch_size)
    print(len(batches))
    for batch in batches:
        print(i)
        dipole = model.apply(
            params,
            atomic_numbers=batch["atomic_numbers"],
            positions=batch["positions"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )
        i += 1
        #lo
        data_final.append(dipole)
    data_final_final = np.vstack(data_final)
    print(data_final_final)

    normas = np.linalg.norm(data_final_final, axis=1)
    
    plt.figure(figsize=(16, 6))
    plt.plot(range(len(normas)), normas, color="b")

    plt.xlabel("Índice del vector")
    plt.ylabel("Norma del vector")
    plt.title("Normas de los vectores en la matriz")
    plt.grid(True)
    plt.savefig("trayectoria.png", dpi=300)  

    plt.show()
    np.savez(f'dipole', data_final_final)