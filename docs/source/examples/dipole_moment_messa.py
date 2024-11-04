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


def magnitude_and_direction_loss(prediction
                                 , target):
    # Calcula la diferencia en magnitud
    magnitude_diff = jnp.abs(jnp.linalg.norm(prediction, axis=-1) - jnp.linalg.norm(target, axis=-1))
    # Calcula el módulo del producto cruz (error direccional)
    cross_product = jnp.cross(prediction, target)
    direction_error = jnp.linalg.norm(cross_product, axis=-1)
    # Combina las pérdidas
    total_loss = magnitude_diff + direction_error
    # Devuelve la pérdida media
    return jnp.mean(total_loss)

def mean_squared_loss(dipole_prediction, dipole_target):
    return jnp.mean(optax.l2_loss(dipole_prediction, dipole_target))

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


def prepare_batches(key, data, batch_size):
    # Determine the number of training steps per epoch.
    data_size = len(data["dipole_moment"])
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
            dipole_moment=data["dipole_moment"][perm].reshape(-1, 3),
            atomic_numbers=atomic_numbers,
            positions=data["positions"][perm].reshape(-1, 3),
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
        )
        for perm in perms
    ]


@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update", "batch_size"))
def train_step(model_apply, optimizer_update, batch, batch_size, opt_state, params):
    def loss_fn(params):
        dipole = model_apply(
            params,
            atomic_numbers=batch["atomic_numbers"],
            positions=batch["positions"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
        )
        #loss = mean_squared_loss(
        #    dipole_prediction=dipole, dipole_target=batch["dipole_moment"]
        #)
        loss =  magnitude_and_direction_loss(dipole
                                 , batch["dipole_moment"])
        return loss

    loss, grad = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@functools.partial(jax.jit, static_argnames=("model_apply", "batch_size"))
def eval_step(model_apply, batch, batch_size, params):
    dipole = model_apply(
        params,
        atomic_numbers=batch["atomic_numbers"],
        positions=batch["positions"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
    )
    print('dipole_prediction',dipole[0])
    print('dipole_target',batch["dipole_moment"][0])
    #loss = mean_squared_loss(
    #    dipole_prediction=dipole, dipole_target=batch["dipole_moment"]
    #)
    loss =  magnitude_and_direction_loss(dipole
                                 , batch["dipole_moment"])
    return loss


def train_model(
    key, model, train_data, valid_data, num_epochs, learning_rate, batch_size
):
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(
        len(train_data["atomic_numbers"])
    )
    params = model.init(
        init_key,
        atomic_numbers=train_data["atomic_numbers"],
        positions=train_data["positions"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    opt_state = optimizer.init(params)

    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)

        # Loop over train batches.
        train_loss = 0.0
        for i, batch in enumerate(train_batches):
            
            params, opt_state, loss = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
            )
            train_loss += (loss - train_loss) / (i + 1)

        # Evaluate on validation set.
        valid_loss = 0.0
        for i, batch in enumerate(valid_batches):
            loss = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=params,
            )
            valid_loss += (loss - valid_loss) / (i + 1)

        # Print progress.
        print(f"epoch: {epoch: 3d}                    train:   valid:")
        print(f"    loss [a.u.]             {train_loss : 8.6f} {valid_loss : 8.3f}")

    # Return final model parameters.
    return params


def train_model(
    key, model, train_data, valid_data, num_epochs, learning_rate, batch_size
):
    # Initialize model parameters and optimizer state.
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(
        len(train_data["atomic_numbers"])
    )
    params = model.init(
        init_key,
        atomic_numbers=train_data["atomic_numbers"],
        positions=train_data["positions"][0],
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    opt_state = optimizer.init(params)

    # Batches for the validation set need to be prepared only once.
    key, shuffle_key = jax.random.split(key)
    valid_batches = prepare_batches(shuffle_key, valid_data, batch_size)

    # Variables to keep track of the best parameters and lowest validation loss.
    best_params = params
    best_valid_loss = float('inf')

    # Train for 'num_epochs' epochs.
    for epoch in range(1, num_epochs + 1):
        # Prepare batches.
        key, shuffle_key = jax.random.split(key)
        train_batches = prepare_batches(shuffle_key, train_data, batch_size)

        # Loop over train batches.
        train_loss = 0.0
        for i, batch in enumerate(train_batches):
            
            params, opt_state, loss = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                batch_size=batch_size,
                opt_state=opt_state,
                params=params,
            )
            train_loss += (loss - train_loss) / (i + 1)

        # Evaluate on validation set.
        valid_loss = 0.0
        for i, batch in enumerate(valid_batches):
            loss = eval_step(
                model_apply=model.apply,
                batch=batch,
                batch_size=batch_size,
                params=params,
            )
            valid_loss += (loss - valid_loss) / (i + 1)

        # Update the best parameters if the current validation loss is lower.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_params = params

        # Print progress.
        print(f"epoch: {epoch: 3d}                    train:   valid:")
        print(f"    loss [a.u.]             {train_loss : 8.6f} {valid_loss : 8.3f}")

    # Return the best model parameters.
    return best_params


def prepare_datasets(filename, key, num_train, num_valid):
    # Load the dataset.
    dataset = np.load(filename)
    num_data = len(dataset["E"])

    Z = jnp.full(1, 23)
    Z = jnp.append(Z, jnp.full(16, 14))

    Z = jnp.expand_dims(Z, axis=0)
    Z = jnp.repeat(Z, num_data, axis=0)
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
    print(valid_choice, 'choicechoicechoicechoice' , choice)
    # Collect and return train and validation sets.
    train_data = dict(
        # energy=jnp.asarray(dataset["E"][train_choice, 0] - mean_energy),
        # forces=jnp.asarray(dataset["F"][train_choice]),
        dipole_moment=jnp.asarray(dataset["D"][train_choice]),
        #atomic_numbers=jnp.asarray(Z[train_choice]),
        atomic_numbers=jnp.asarray(dataset["z"]),
        # atomic_numbers=jnp.asarray(z_hack),
        positions=jnp.asarray(dataset["R"][train_choice]),
    )
    valid_data = dict(
        # energy=jnp.asarray(dataset["E"][valid_choice, 0] - mean_energy),
        # forces=jnp.asarray(dataset["F"][valid_choice]),
        #atomic_numbers=jnp.asarray(Z[valid_choice]),
        dipole_moment=jnp.asarray(dataset["D"][valid_choice]),
        # atomic_numbers=jnp.asarray(z_hack),
        atomic_numbers=jnp.asarray(dataset["z"]),
        positions=jnp.asarray(dataset["R"][valid_choice]),
    )
    return train_data, valid_data

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

num_train = 5000
num_val = 1000
# Define training hyperparameters.
learning_rate = 0.001
num_epochs = 3000
batch_size = 512
# Model hyperparameters.
features = 128
max_degree = 2
num_iterations = 3
num_basis_functions = 16
cutoff = 6.0
max_atomic_number = 23
model_save_path = "mode_training_Si16Vplus..DFT.SP-GRD.wB97X-D.tight.Data.5986.R_E_F_D_Q.pkl"
filename = '../e3x_tranfer/Si16Vplus..DFT.SP-GRD.PBE.tight.Data.6000.R_E_F_D_Q.npz'
filename_trayec = "../e3x_tranfer/docs/source/examples/SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_400K_1B_01_POSITION_0_reshape.npz"

filename_trayec_2 = "../e3x_tranfer/SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_100K_1B_01_POSITION_0.npz"
filename_trayec_2_reshape = "../e3x_tranfer/SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_100K_1B_01_POSITION_0_reshape.npz"

if __name__ == "__main__":


    
    dataset = np.load(filename)
    for key in dataset.keys():
        print(key)
    print("Dipole moment shape array", dataset["D"].shape)
    print("Dipole moment units", dataset["D_units"])

    print("Atomic numbers", dataset["z"])
    print("R shape array", dataset["R"].shape)
    key = jax.random.PRNGKey(0)

    train_data, valid_data = prepare_datasets(filename, key, num_train, num_val)
    key, train_key = jax.random.split(key)
    model = MessagePassingModel(
        features=features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
        max_atomic_number=max_atomic_number
    )
    params = train_model(
        key=train_key,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    
    
    with open(model_save_path, "wb") as f:
        pickle.dump(params, f)

    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(valid_data["atomic_numbers"]))

    predict = []

    for i in range(len(valid_data["positions"])) : 
        dm=model.apply(params,
            atomic_numbers=valid_data["atomic_numbers"],
            positions=valid_data["positions"][i],
            dst_idx=dst_idx,
            src_idx=src_idx)
        predict.append(dm)

    result = np.vstack(predict)

    total = 0
    list_angle = []
    to = 0
    for i in range(len((predict))):
        v = predict[i][0]
        u = valid_data['dipole_moment'][i]
        angle_degrees = angle_between(v, u)
        list_angle.append(angle_degrees)
        #if angle_degrees < 80 : 
        #    to += 1
        total += angle_degrees**2

    print(np.sqrt(total / len(valid_data['dipole_moment'])))
    #print(predict)

    print("mean_absolute_error :", mean_absolute_error(valid_data['dipole_moment'], result))
    print("mean squared error", jnp.mean((result - valid_data['dipole_moment']) ** 2))
    print("RMSE ", root_mean_squared_error(valid_data['dipole_moment'], result))

    plt.hist(
    list_angle,
    bins=80,
    edgecolor="black",
    )

    plt.xlabel("Ángulo (°)")
    plt.ylabel("Densidad de Frecuencia")
    plt.title("Histograma de la Distribución de Ángulos")
    plt.savefig("histograma_angulos.png", dpi=300) 

    plt.show()


    ################################################load model and predict
    filename_trayec_2 = "../e3x_tranfer/SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_100K_1B_01_POSITION_0.npz"
    model_save_path = "mode_training_Si16Vplus..DFT.SP-GRD.wB97X-D.tight.Data.5986.R_E_F_D_Q.pkl"

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
    
    '''    dataset = np.load(filename_trayec_2, allow_pickle=True)
        for key in dataset.keys():
            print(key)

        print("R", dataset["R"].shape)
        # Modificar el array "R"
        dataset_modified = {
            key: np.squeeze(value, axis=0) if key == "R" else value
            for key, value in dataset.items()
        }

        # Guardar el dataset modificado
        np.savez(
            filename_trayec_2_reshape
            **dataset_modified,
            allow_pickle=False,
        )
        dataset = np.load(filename_trayec_2_reshape, allow_pickle=False)
        for key in dataset.keys():
            print(key)
    '''
    
    trayect= prepare_trayect(
        filename_trayec_2_reshape
    )
    data_final = []
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(trayect["atomic_numbers"][0]))

    i = 0
    batch_size = 512
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
    # Graficar las normas
    plt.figure(figsize=(16, 6))
    plt.plot(range(len(normas)), normas, color="b")

    # Etiquetas y título
    plt.xlabel("Índice del vector")
    plt.ylabel("Norma del vector")
    plt.title("Normas de los vectores en la matriz")
    plt.grid(True)
    plt.savefig("trayectoria.png", dpi=300)  # Guarda el archivo como PNG con 300 DPI
    # Mostrar la gráfica
    plt.show()
    np.savez('SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_400K_1B_01_POSITION_0_dipole.npz', data_final_final)