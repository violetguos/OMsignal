import numpy as np
import torch
import itertools
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))   # Hack - fix package management later

from torch.utils.data import DataLoader, Dataset
from src.legacy.TeamB1pomt5.code.config import DATA_DIR, CLUSTER_DIR
from src.legacy.TeamB1pomt5.code.omsignal.utils.memfile_utils import read_memfile
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
import warnings
warnings.filterwarnings('ignore')

class OM_dataset(Dataset):
    '''
    Actual dataset class for non-ranking tasks.
    '''
    def __init__(self, sequences, targets, transform=None):
        self.sequences_preproc = sequences
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.sequences_preproc)

    def __getitem__(self, index):
        sequence_preproc, targets = self.sequences_preproc[index], self.targets[index]
        if self.transform is not None:
            sequence_preproc = self.transform(sequence_preproc)  # Transform the preprocessed data, not original
        return sequence_preproc, targets

class Rank_dataset(Dataset):
    '''
    Dataset class for ranking tasks.
    '''
    def __init__(self, sequences, targets, transform=None):
        # This version takes data in that is already preprocessed
        self.sequences_preproc = sequences
        self.targets = targets
        self.transform = transform

        # Get all pairwise non-repeated combinations (cartesian product)
        seq_length = self.sequences_preproc.shape[0]
        combos = list(itertools.product(range(seq_length), range(seq_length)))
        idx_cart_product = [(a,b) for a, b in combos if a != b]

        # Create new inputs that stack the items in the pair on top of each other
        self.compare_input = np.zeros((len(idx_cart_product), 1, 2, 3750))
        self.compare_target = np.zeros((len(idx_cart_product), 3))
        for idx, (i,j) in enumerate(idx_cart_product):
            self.compare_input[idx] = np.vstack((self.sequences_preproc[i,:], self.sequences_preproc[j,:]))
            # changed < to > so it makes more sense (1 if bigger, 0 if smaller)
            self.compare_target[idx] = self.targets[i, 0:3] > self.targets[j, 0:3]

    def __len__(self):
        return len(self.sequences_preproc)

    def __getitem__(self, index):
        sequence, target = self.compare_input[index], self.compare_target[index]
        if self.transform is not None:
            sequence = self.transform(sequence)

        return sequence, target

def import_OM(dataset, cluster=False):
    '''
    Reads in the data from the original memmap, separating by type.
    '''
    dataset = dataset.capitalize() # 'Train', 'Validation'
    if cluster:
        train_path = os.path.join(CLUSTER_DIR, 'MILA_{}LabeledData.dat'.format(dataset))
    else:
        train_path = os.path.join(DATA_DIR, 'MILA_{}LabeledData_dummy.dat'.format(dataset))
    
    data = read_memfile(train_path, shape=(160, 3754), dtype='float32')

    # Split the data according to what's in the columns
    ecg = data[:, :3750]
    ecg = ecg.reshape(ecg.shape[0],1,ecg.shape[1])
    PR_means = data[:, 3750].reshape((-1,1))
    RT_means = data[:, 3751].reshape((-1,1))
    RR_stdevs = data[:, 3752].reshape((-1,1))
    ids = data[:, 3753].reshape((-1,1))

    return ecg, PR_means, RT_means, RR_stdevs, ids

def import_train_valid(y, cluster=False):
    '''
    Returns data split into training/validation sets, according to specified y.
    y can be: PR_means, RT_means, RR_stdevs, ids
    '''
    ecg_train, PR_means_train, RT_means_train, \
        RR_stdevs_train, ids_train = import_OM('train', cluster=cluster)
    ecg_valid, PR_means_valid, RT_means_valid, \
        RR_stdevs_valid, ids_valid = import_OM('validation', cluster=cluster)

    if y not in ['PR_means', 'RT_means', 'RR_stdevs', 'ids', 'all']:
            raise ValueError("y must be in ['PR_means', 'RT_means', 'RR_stdevs', 'ids', 'all']")
    y_data = {'PR_means' : [PR_means_train, PR_means_valid],
              'RT_means' : [RT_means_train, RT_means_valid],
              'RR_stdevs' : [RR_stdevs_train, RR_stdevs_valid],
              'ids' : [ids_train, ids_valid]}
    if y == 'all': 
        y_train = np.hstack((PR_means_train, RT_means_train, RR_stdevs_train, ids_train))
        y_valid = np.hstack((PR_means_valid, RT_means_valid, RR_stdevs_valid, ids_valid))
    else:
        y_train = y_data[y][0]
        y_valid = y_data[y][1]
    return ecg_train, ecg_valid, y_train, y_valid

def get_dataloader(X, y, transform, batch_size=50, shuffle=True, task_type = "Regression"):
    if task_type not in ["Regression", "Classification" , "Ranking"]:
        raise ValueError("task_type must be in ['Regression', 'Classification' , 'Ranking']")

    if task_type == "Ranking":
        dataset = Rank_dataset(X, y, transform=transform)
    elif task_type == "Classification":
        # X is n,1,3750, we want to calculate the FFT over the last axis
        arr_copy = np.copy(X)
        arr_copy = np.fft.rfft(arr_copy, axis=2).astype(np.float32)
        dataset = OM_dataset(arr_copy, y, transform=transform)    
    else:
        dataset = OM_dataset(X, y, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader


"""
OLD functions

def multitask_train_valid(cluster=False):
    '''
    Creates training/validation set for all ys by grouping ys into
    [PR, RT, RR, id].
    '''
    X_train, X_valid, y_train, y_valid = import_train_valid('all', cluster=cluster)
    y_PR_train, y_PR_valid = y_train[0], y_valid[0]
    y_RT_train, y_RT_valid = y_train[1], y_valid[1]
    y_RR_train, y_RR_valid = y_train[2], y_valid[2]
    y_ids_train, y_ids_valid = y_train[3], y_valid[3]

    # Training
    stacked = np.vstack((y_PR_train, y_RT_train, y_RR_train, y_ids_train))
    y_train = stacked.T

    # Validation
    stacked = np.vstack((y_PR_valid, y_RT_valid, y_RR_valid, y_ids_valid))
    y_valid = stacked.T

    return X_train, X_valid, y_train, y_valid

def make_tensor_fft(x):
    '''
    Wraps make_fft to handle tensor issues. Only returns real component.
    Input is a torch tensor containing a batch of data.
    '''
    try:
        x = x.cpu().numpy()
    except:
        x = x.detach().cpu().numpy()
    x_fft = np.array([make_fft(i) for i in x])
    x_fft = torch.from_numpy(x_fft)
    return x_fft.to(device)

def make_fft(x, imag=False):
    # takes a time-series x and returns the FFT
    # input: x
    # output : R and I; real and imaginary componenet of the real FFT
    y = np.fft.rfft(x)
    y = y.astype(np.float32)    # Otherwise pytorch will cast float64's to double tensors
    if imag:
        return np.real(y), np.imag(y)
    else:
        return np.real(y)
"""

