import functools
import e3x
from flax import linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# Disable future warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_datasets(filename, key, num_train, num_valid):
    # Load the dataset.
    dataset = np.load(filename)
    num_data = len(dataset["E"])
    Z=jnp.full(16,14)
    Z=jnp.append(Z,23)
    Z=jnp.expand_dims(Z,axis=0)
    Z=jnp.repeat(Z, num_data, axis=0)
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
        #energy=jnp.asarray(dataset["E"][train_choice, 0] - mean_energy),
        #forces=jnp.asarray(dataset["F"][train_choice]),
        dipole_moment= jnp.asarray(dataset["D"][train_choice]),
        atomic_numbers=jnp.asarray(Z[train_choice]),
        # atomic_numbers=jnp.asarray(z_hack),
        positions=jnp.asarray(dataset["R"][train_choice]),
    )
    valid_data = dict(
        #energy=jnp.asarray(dataset["E"][valid_choice, 0] - mean_energy),
        #forces=jnp.asarray(dataset["F"][valid_choice]),
        atomic_numbers=jnp.asarray(Z[valid_choice]),
        dipole_moment= jnp.asarray(dataset["D"][valid_choice]),
        # atomic_numbers=jnp.asarray(z_hack),
        positions=jnp.asarray(dataset["R"][valid_choice]),
    )
    return train_data, valid_data


class Dipole_Moment(nn.Module):
  #features = 1
  #max_degree = 1
  @nn.compact
  def __call__(self,atomic_numbers, positions):  # Shapes (..., N) and (..., N, 3).
    # 1. Initialize features.
    x = jnp.concatenate((atomic_numbers[...,None], positions), axis=-1) # Shape (..., N, 4).
    #print("Initial shape:", x.shape)
    x = x[..., None, :, None]  # Shape (..., N, 1, 3, 1).
    #print("x shape:", x.shape)
    # 2. Apply transformations.
    x = e3x.nn.Dense(features=1)(x) 
    #print("After Dense layer:", x.shape)
    x = e3x.nn.TensorDense(max_degree=1)(x)  
    #print("After TensorDense layer:", x.shape)
    x=jnp.sum(x, axis=-4) 
    #print("After sum:", x.shape)
    y = x[..., 1, 1:4, 0]
    #print("After slicing:", y.shape)

    return y
  

def mean_squared_loss(prediction, target):
  return jnp.mean(optax.l2_loss(prediction, target))


@functools.partial(jax.jit, static_argnames=('model_apply', 'optimizer_update'))
def train_step(model_apply, optimizer_update, batch, opt_state, params):
  def loss_fn(params):
    inertia_tensor = model_apply(params, batch['masses'], batch['positions'])
    loss = mean_squared_loss(inertia_tensor, batch['inertia_tensor'])
    return loss
  loss, grad = jax.value_and_grad(loss_fn)(params)
  updates, opt_state = optimizer_update(grad, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss

@functools.partial(jax.jit, static_argnames=('model_apply',))
def eval_step(model_apply, batch, params):
  inertia_tensor = model_apply(params, batch['masses'], batch['positions'])
  loss = mean_squared_loss(inertia_tensor, batch['inertia_tensor'])
  return loss

def train_model(key, model, train_data, valid_data, num_epochs, learning_rate, batch_size):
  # Initialize model parameters and optimizer state.
  key, init_key = jax.random.split(key)
  optimizer = optax.adam(learning_rate)
  params = model.init(init_key, train_data['masses'][0:1], train_data['positions'][0:1])
  opt_state = optimizer.init(params)

  # Determine the number of training steps per epoch.
  train_size = len(train_data['masses'])
  steps_per_epoch = train_size//batch_size

  # Train for 'num_epochs' epochs.
  for epoch in range(1, num_epochs + 1):
    # Draw random permutations for fetching batches from the train data.
    key, shuffle_key = jax.random.split(key)
    perms = jax.random.permutation(shuffle_key, train_size)
    perms = perms[:steps_per_epoch * batch_size]  # Skip the last batch (if incomplete).
    perms = perms.reshape((steps_per_epoch, batch_size))

    # Loop over all batches.
    train_loss = 0.0  # For keeping a running average of the loss.
    for i, perm in enumerate(perms):
      batch = {k: v[perm, ...] for k, v in train_data.items()}
      params, opt_state, loss = train_step(
          model_apply=model.apply,
          optimizer_update=optimizer.update,
          batch=batch,
          opt_state=opt_state,
          params=params
      )
      train_loss += (loss - train_loss)/(i+1)

    # Evaluate on the test set after each training epoch.
    valid_loss = eval_step(
        model_apply=model.apply,
        batch=valid_data,
        params=params
    )

    # Print progress.
    print(f"epoch {epoch : 4d} train loss {train_loss : 8.6f} valid loss {valid_loss : 8.6f}")

  # Return final model parameters.
  return params


filename='Si16Vplus..DFT.SP-GRD.wB97X-D.tight.Data.5042.R_E_F_D_Q.npz'
# Initialize PRNGKey for random number generation.
key = jax.random.PRNGKey(0)
num_train=3000
num_val=200
train_data, valid_data = prepare_datasets(filename,key, num_train,num_val)

# Define training hyperparameters.
learning_rate = 0.002
num_epochs = 100
batch_size = 32

key, train_key = jax.random.split(key)
model = Dipole_Moment()
params = train_model(
  key=train_key,
  model=model,
  train_data=train_data,
  valid_data=valid_data,
  num_epochs=num_epochs,
  learning_rate=learning_rate,
  batch_size=batch_size,
)