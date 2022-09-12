import copy
import scipy.io  # to read .mat files
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
import torch_geometric
from config import Config


DATA_PATH = "/home/latis/Documents/RegGNN/IQ_data_clean/"  # this should point to the directory of connectomes in .mat format


def load_matrices_from_matfile(pop="NT"):
    '''Loads the matrices for given population, for 226 subjects
    Args:
        pop (string): Population code, "NT" or "ASD"

    Returns:
        connectomes (np.array): Connectome tensor of shape (116x116x226)
        fiq_scores (np.array): FIQ score vector of shape (226x1)
        viq_scores (np.array): FIQ score vector of shape (226x1)

    Raises:
        ValueError: if given population is not "NT" or "ASD"
    '''
    if pop not in ["NT", "ASD"]:
        raise ValueError("Population not found")
    connectomes = scipy.io.loadmat(f"{DATA_PATH}/allMatrices_{pop}.mat")[f"allMatrices_{pop}"]
    fiq_scores = scipy.io.loadmat(f"{DATA_PATH}/FIQ_{pop}.mat")[f"FIQ_{pop}"]
    viq_scores = scipy.io.loadmat(f"{DATA_PATH}/VIQ_{pop}.mat")[f"VIQ_{pop}"]

    connectomes = torch.tensor(np.nan_to_num(connectomes))
    fiq_scores = torch.tensor(fiq_scores)
    viq_scores = torch.tensor(viq_scores)
    return connectomes, fiq_scores, viq_scores


def create_dataset():
    '''Does preprocessing on matrices in .mat files and saves them to pickle files

    Files will be saved in connectome_{population}.pickle, fiq_{population}.pickle,
    and viq_{population}.pickle

    '''
    con_n, fiq_n, viq_n = load_matrices_from_matfile("NT")
    con_a, fiq_a, viq_a = load_matrices_from_matfile("ASD")

    con_n[con_n < 0] = 0
    con_a[con_a < 0] = 0

    torch.save(con_n, f"{Config.DATA_FOLDER}connectome_NT.ts")
    torch.save(fiq_n, f"{Config.DATA_FOLDER}fiq_NT.ts")
    torch.save(viq_n, f"{Config.DATA_FOLDER}viq_NT.ts")

    torch.save(con_a, f"{Config.DATA_FOLDER}connectome_ASD.ts")
    torch.save(fiq_a, f"{Config.DATA_FOLDER}fiq_ASD.ts")
    torch.save(viq_a, f"{Config.DATA_FOLDER}viq_ASD.ts")


def load_dataset_pytorch(pop="ASD", score="fiq"):
    '''Loads the data for the given population into a list of Pytorch Geometric
    Data objects, which then can be used to create DataLoaders.
    '''
    connectomes = torch.load(f"{Config.DATA_FOLDER}connectome_{pop}.ts")
    scores = torch.load(f"{Config.DATA_FOLDER}{score}_{pop}.ts")

    pyg_data = []
    for subject in range(scores.shape[0]):
        sparse_mat = to_sparse(connectomes[:, :, subject])
        pyg_data.append(torch_geometric.data.Data(x=torch.eye(116, dtype=torch.float),
                                                  y=scores[subject].float(), edge_index=sparse_mat._indices(),
                                                  edge_attr=sparse_mat._values().float()))

    return pyg_data


def to_sparse(mat):
    '''Transforms a square matrix to torch.sparse tensor

    Methods ._indices() and ._values() can be used to access to
    edge_index and edge_attr while generating Data objects
    '''
    coo = coo_matrix(mat, dtype='float64')
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    coo_index = torch.stack([row, col], dim=0)
    coo_values = torch.from_numpy(coo.data.astype(np.float64).reshape(-1, 1)).reshape(-1)
    sparse_mat = torch.sparse.LongTensor(coo_index, coo_values)
    return sparse_mat


def load_dataset_cpm(pop="NT"):
    '''Loads the data for given population in the upper triangular matrix form
    as required by CPM functions.
    '''
    connectomes = np.array(torch.load(f"connectome_{pop}.ts"))
    fiq_scores = np.array(torch.load(f"fiq_{pop}.ts"))
    viq_scores = np.array(torch.load(f"viq_{pop}.ts"))

    fc_data = {}
    behav_data = {}
    for subject in range(fiq_scores.shape[0]):  # take upper triangular part of each matrix
        fc_data[subject] = connectomes[:, :, subject][np.triu_indices_from(connectomes[:, :, subject], k=1)]
        behav_data[subject] = {'fiq': fiq_scores[subject].item(), 'viq': viq_scores[subject].item()}
    return pd.DataFrame.from_dict(fc_data, orient='index'), pd.DataFrame.from_dict(behav_data, orient='index')


def get_folds(data_list, k_folds=5):
    '''Divides a data list into lists
       with k elements such that each element
       is the data used in that cross validation fold
    '''
    train_folds, test_folds = [], []
    for train_idx, test_idx in KFold(k_folds, shuffle=False, random_state=None).split(data_list):
        train_folds.append([data_list[i] for i in train_idx])
        test_folds.append([data_list[i] for i in test_idx])
    return train_folds, test_folds


def get_loaders(train, test, batch_size=1):
    '''Returns data loaders for given data lists
    '''
    train_loader = torch_geometric.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch_geometric.data.DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def load_dataset_tensor(pop="NT"):
    '''Loads dataset as tuple of (tensor of connectomes,
       tensor of fiq scores, tensor of viq scores)
    '''
    connectomes = torch.load(f"connectome_{pop}.ts")
    fiq_scores = torch.load(f"fiq_{pop}.ts")
    viq_scores = torch.load(f"viq_{pop}.ts")
    return connectomes, fiq_scores, viq_scores


def to_dense(data):
    '''Returns a copy of the data object in Dense form.
    '''
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data