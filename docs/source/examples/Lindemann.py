import time

import numpy as np
import torch


class get_Lindemann_gpu:
    def __init__(self, mol_structures_th):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Working on:", self.device)
        self.mol_structures_th = mol_structures_th.to(self.device)
        self.n_molecules = mol_structures_th.shape[0]
        self.n_atoms = mol_structures_th.shape[-2]

    def get_distances(self):
        print("Computing distances...")
        rij_tensor = torch.cdist(self.mol_structures_th, self.mol_structures_th)
        print("rij_tensor.shape", rij_tensor.shape)
        return rij_tensor

    def get_Lindemann_index(self):
        start_time = time.time()

        # Compute distances
        rij_tensor = self.get_distances()

        # Lindemann index calculations
        mean_rij2 = torch.mean(rij_tensor**2, dim=1)
        mean_rij = torch.mean(rij_tensor, dim=1)
        print("mean_rij.shape", mean_rij.shape)

        # Mask to get just the triangular part of the matrix
        mask = (
            torch.triu(
                torch.ones(self.n_atoms, self.n_atoms, device=self.device), diagonal=1
            )
            == 1
        )
        mask = mask.expand(self.n_molecules, self.n_atoms, self.n_atoms)
        print("mask.shape", mask.shape)
        print(
            "torch.triu.shape",
            torch.reshape(mean_rij[mask], (self.n_molecules, -1)).shape,
        )

        # Reshape and compute delta_i_ij
        mean_rij2 = torch.reshape(mean_rij2[mask], (self.n_molecules, -1))
        mean_rij = torch.reshape(mean_rij[mask], (self.n_molecules, -1))
        delta_i_ij = torch.div(torch.sqrt(mean_rij2 - mean_rij**2), mean_rij2)

        # Final delta computation
        delta = 2 * torch.sum(delta_i_ij, dim=1) / (self.n_atoms * (self.n_atoms - 1))

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000  # time in milliseconds
        print(f"Time taken: {elapsed_time_ms:.4f} ms")

        return delta, torch.mean(delta), torch.std(delta)


filename = "/home/beemoqc2/Documents/e3x_tranfer/docs/source/examples/test_data.npz"
dataset = np.load(filename)
dataset["R"].shape


filename = "/home/beemoqc2/Documents/e3x_tranfer/SI16VPLUS_E3X_RETRAINED_WB97X_D_TIGHT_TRP_400K_1B_01_POSITION_0.npz"
dataset = np.load(filename)
dataset = np.squeeze(dataset["R"], axis=0)
print("dataset shape:", dataset.shape)

for key in dataset.keys():
    print(key)
T_1 = dataset[0:50000]
T_1 = np.expand_dims(T_1, axis=1)
T_2 = dataset[50000:100000]
T_2 = np.expand_dims(T_2, axis=1)
T_3 = dataset[100000:150000]
T_3 = np.expand_dims(T_3, axis=1)
T_4 = dataset[150000:200000]
T_4 = np.expand_dims(T_4, axis=1)


molecule = np.concatenate((T_1, T_2, T_3, T_4), axis=1)
print("tensor molecule shape:", molecule.shape)


def split_and_concatenate(dataset, num_splits):
    # Determinar el tamaño de cada split
    split_size = dataset.shape[0] // num_splits

    # Lista para almacenar los splits
    T_list = []

    for i in range(num_splits):
        # Crear cada split
        T_i = dataset[i * split_size : (i + 1) * split_size]
        # Expandir la dimensión
        T_i = np.expand_dims(T_i, axis=1)
        # Agregar a la lista
        T_list.append(T_i)

    # Concatenar los splits a lo largo del eje 1
    molecule = np.concatenate(T_list, axis=1)

    return molecule


num_splits = 100  # Por ejemplo, 4
molecule = split_and_concatenate(dataset, num_splits)

print("tensor molecule shape:", molecule.shape)

lindemann_index = get_Lindemann_gpu(torch.as_tensor(molecule))
delta, mean_delta, std_delta = lindemann_index.get_Lindemann_index()

delta.shape

print("Lindemann index:", delta)
print("Mean Lindemann index:", mean_delta)
print("Standard deviation of Lindemann index:", std_delta)

# Convertir el tensor a un array de numpy para graficar
tensor_numpy = delta.cpu()

import matplotlib.pyplot as plt

# Graficar el tensor
plt.plot(tensor_numpy)
plt.title("Gráfico de un Tensor 1D")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.grid(True)
plt.show()
