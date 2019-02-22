import numpy as np
import torch
import itertools
import os
import sys
#sys.path.append(os.path.abspath(os.path.join('..')))   # Hack - fix package management later

from src.legacy.TeamB1pomt5.code.config import DATA_DIR, CLUSTER_DIR
from src.legacy.TeamB1pomt5.code.omsignal.utils.memfile_utils import read_memfile
from torch.utils.data import DataLoader, Dataset

def import_OM(dataset, cluster=True): # Ideally replacing in legacy code : single function to import either unlabeled or labeled
    """
    Reads in the unlabeled data from the original memmap, separating by type.

    :param dataset: (string) dataset type : train, valid, unlabeled
    :param cluster: (bool) location of the data (in case of dummy)
    :return: desired dataset
    """
    labeled = False
    switch = {"Train": "TrainLabeled", "train": "TrainLabeled", "valid": "ValidationLabeled", "Valid": "ValidationLabeled"}
    dataset = dataset.capitalize()  # 'Train', 'Validation'
    if dataset in ["Train", "Validation", "Training", "Valid"]:
        dataset_ = switch[dataset]
        labeled = True

    if cluster:
        path_ = os.path.join(CLUSTER_DIR, 'MILA_{}Data.dat'.format(dataset))
    else:
        path_ = os.path.join(DATA_DIR, 'MILA_{}Data_dummy.dat'.format(dataset))

    if labeled:
        shape_ = (160, 3754)
    else:
        shape_ = (657233, 3750)

    data = read_memfile(path_, shape=shape_, dtype='float32')

    # Split the data according to what's in the columns
    if labeled:
        ecg = data[:, :3750]
        ecg = ecg.reshape(ecg.shape[0], 1, ecg.shape[1])
        PR_means = data[:, 3750].reshape((-1, 1))
        RT_means = data[:, 3751].reshape((-1, 1))
        RR_stdevs = data[:, 3752].reshape((-1, 1))
        ids = data[:, 3753].reshape((-1, 1))

        return ecg, PR_means, RT_means, RR_stdevs, ids
    else:
        return data

def entropy_mnimization_loss():

if __name__ == '__main__':

    data_ = import_OM("unlabeled")
    print(np.shape(data_))