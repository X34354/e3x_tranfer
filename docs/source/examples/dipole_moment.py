import functools
import pickle

# Disable future warnings.
import warnings
from datetime import datetime
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optax
import plotly.graph_objs as go
import plotly.io as pio
from flax import linen as nn

import e3x

warnings.simplefilter(action="ignore", category=FutureWarning)


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


def prepare_datasets(filename, key, num_train, num_valid):
    # Load the dataset.
    dataset = np.load(filename)
    for j in range(len(dataset["R"])):
        dataset["R"][j, :, :] = dataset["R"][j, :, :] - dataset["R"][j, 0, :]
    num_data = len(dataset["E"])
    Z = jnp.full(16, 14)
    Z = jnp.insert(Z, 0, 23)
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

    # Collect and return train and validation sets.
    train_data = dict(
        # energy=jnp.asarray(dataset["E"][train_choice, 0] - mean_energy),
        # forces=jnp.asarray(dataset["F"][train_choice]),
        dipole_moment=jnp.asarray(dataset["D"][train_choice]),
        atomic_numbers=jnp.asarray(Z[train_choice]),
        # atomic_numbers=jnp.asarray(z_hack),
        positions=jnp.asarray(dataset["R"][train_choice]),
    )
    valid_data = dict(
        # energy=jnp.asarray(dataset["E"][valid_choice, 0] - mean_energy),
        # forces=jnp.asarray(dataset["F"][valid_choice]),
        atomic_numbers=jnp.asarray(Z[valid_choice]),
        dipole_moment=jnp.asarray(dataset["D"][valid_choice]),
        # atomic_numbers=jnp.asarray(z_hack),
        positions=jnp.asarray(dataset["R"][valid_choice]),
    )
    return train_data, valid_data


class Dipole_Moment(nn.Module):
    features = 8

    # max_degree = 1
    @nn.compact
    def __call__(self, atomic_numbers, positions):  # Shapes (..., N) and (..., N, 3).
        # 1. Initialize features.
        x = jnp.concatenate(
            (atomic_numbers[..., None], positions), axis=-1
        )  # Shape (..., N, 4).
        # print("Initial shape:", x.shape)
        x = x[..., None, :, None]  # Shape (..., N, 1, 3, 1).
        # print("x shape:", x.shape)
        # 2. Apply transformations.
        x = e3x.nn.Dense(self.features)(x)
        # print("After Dense layer:", x.shape)
        x = e3x.nn.TensorDense(features=1, max_degree=1)(x)
        # print("After TensorDense layer:", x.shape)
        x = jnp.sum(x, axis=-4)
        # print("After sum:", x.shape)
        y = x[..., 1, 1:4, 0]
        # print("After slicing:", y.shape)

        return y


def mean_squared_loss(prediction, target):
    return jnp.mean(optax.l2_loss(prediction, target))


@functools.partial(jax.jit, static_argnames=("model_apply", "optimizer_update"))
def train_step(model_apply, optimizer_update, batch, opt_state, params):
    def loss_fn(params):
        dipole_moment = model_apply(params, batch["atomic_numbers"], batch["positions"])
        loss = mean_squared_loss(dipole_moment, batch["dipole_moment"])
        return loss

    loss, grad = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@functools.partial(jax.jit, static_argnames=("model_apply",))
def eval_step(model_apply, batch, params):
    dipole_moment = model_apply(params, batch["atomic_numbers"], batch["positions"])
    loss = mean_squared_loss(dipole_moment, batch["dipole_moment"])
    return loss


def train_model(
    key, model, train_data, valid_data, num_epochs, learning_rate, batch_size
):
    """
    Train a molecular property prediction model using JAX.

    Args:
        key (jax.random.PRNGKey): A JAX random key for deterministic behavior in stochastic processes.
        model (flax.linen.Module): A neural network model defined with Flax linen.
        train_data (dict): A dictionary containing training dataset features such as 'atomic_numbers' and 'positions'.
        valid_data (dict): A dictionary containing validation dataset features.
        num_epochs (int): The number of epochs to train the model.
        learning_rate (float): The learning rate for the Adam optimizer.
        batch_size (int): The number of examples in each batch.

    Returns:
        tuple: A tuple containing the final model parameters, training loss list, and validation loss list.
    """
    # Initialize model parameters and optimizer.
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(learning_rate)
    params = model.init(
        init_key, train_data["atomic_numbers"][0:1], train_data["positions"][0:1]
    )
    opt_state = optimizer.init(params)

    # Calculate the number of batches per epoch.
    train_size = len(train_data["positions"])
    steps_per_epoch = train_size // batch_size

    list_train_loss = []
    list_val_loss = []
    best_val_loss = float("inf")
    best_params = None
    number_epoch = 0
    epochs_no_improve = 0
    patience = 1000  # Number of epochs to wait for improvement

    for epoch in range(1, num_epochs + 1):
        key, shuffle_key = jax.random.split(key)
        perms = jax.random.permutation(shuffle_key, train_size)
        perms = perms[
            : steps_per_epoch * batch_size
        ]  # Ensure all batches have the same size.
        perms = perms.reshape((steps_per_epoch, batch_size))

        train_loss = 0.0

        for i, perm in enumerate(perms):
            batch = {k: v[perm, ...] for k, v in train_data.items()}
            params, opt_state, loss = train_step(
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                batch=batch,
                opt_state=opt_state,
                params=params,
            )
            train_loss += (loss - train_loss) / (i + 1)

        valid_loss = eval_step(model_apply=model.apply, batch=valid_data, params=params)
        list_train_loss.append(train_loss)
        list_val_loss.append(valid_loss)

        # Update best validation loss and save parameters if current loss is lower.
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_train_loss = train_loss
            best_params = params
            number_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"epoch {epoch : 4d} train loss {train_loss : 8.6f} valid loss {valid_loss : 8.6f}"
        )

        # Early stopping if no improvement after `patience` epochs.
        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch} with best validation loss {best_val_loss : 8.6f}"
            )
            break

    # Optionally return the best_params if you want to use the model with the lowest validation loss.
    return (
        best_params,
        list_train_loss,
        list_val_loss,
        number_epoch,
        best_val_loss,
        best_train_loss,
    )


filename = "test_data.npz"
experiment_name = "dipole_moment"

num_train = 4000
num_val = 1000
# Define training hyperparameters.
learning_rate = 0.002
num_epochs = 5000
batch_size = 516
test_value = True
if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    train_data, valid_data = prepare_datasets(filename, key, num_train, num_val)
    key, train_key = jax.random.split(key)

    model = Dipole_Moment()

    (
        best_params,
        list_train_loss,
        list_val_loss,
        number_epoch,
        best_val_loss,
        best_train_loss,
    ) = train_model(
        key=train_key,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        mlflow.log_param("dataset", filename)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("num_train", num_train)
        mlflow.log_param("num_val", num_val)
        mlflow.log_param("best epoch", number_epoch)

        mlflow.log_metric("best_train_loss", best_train_loss)
        mlflow.log_metric("best_val_loss", best_val_loss)

        model_save_path = "best_model_params_test.pkl"
        with open(model_save_path, "wb") as f:
            pickle.dump(best_params, f)

        mlflow.log_artifact(model_save_path)

        create_loss_plot(
            list_train_loss,
            list_val_loss,
            "Training Loss",
            "Validation Loss",
            "Training vs Validation Loss (Train)",
            "train_vs_val_train_dipole_moment.html",
        )
        mlflow.end_run()

    if test_value:

        i = 45
        Z, positions, target = (
            valid_data["atomic_numbers"][i],
            valid_data["positions"][i],
            valid_data["dipole_moment"][i],
        )

        prediction = model.apply(best_params, Z, positions)

        print("target")
        print(target)
        print("prediction")
        print(prediction)
        print("mean squared error", jnp.mean((prediction - target) ** 2))
