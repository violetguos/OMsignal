import numpy as np
import os

from src.legacy.TeamB1pomt5.code.config import DATA_DIR, CLUSTER_DIR
from src.legacy.TeamB1pomt5.code.omsignal.utils.memfile_utils import read_memfile


def import_OM(dataset, cluster=True, len=50000): # Ideally replacing in legacy code : single function to import either unlabeled or labeled
    """
    Reads in the unlabeled data from the original memmap, separating by type.

    :param dataset: (string) dataset type : train, valid, unlabeled
    :param cluster: (bool) location of the data (in case of dummy)
    :param len: (int) amount of unlabeled example to import
    :return: desired dataset
    """
    labeled = False
    switch = {"Train": "TrainLabeled", "train": "TrainLabeled", "Training":"TrainLabeled",
              "valid": "ValidationLabeled", "Valid": "ValidationLabeled", "Validation": "ValidationLabeled"}
    dataset = dataset.capitalize()  # 'Train', 'Validation'
    if dataset in ["Train", "Validation", "Training", "Valid"]:
        dataset_ = switch[dataset]
        labeled = True
    elif dataset in ["Unlabeled"]:
        dataset_ = dataset

    if cluster:
        path_ = os.path.join(CLUSTER_DIR, 'MILA_{}Data.dat'.format(dataset_))
    else:
        path_ = os.path.join(DATA_DIR, 'MILA_{}Data_dummy.dat'.format(dataset_))

    if labeled:
        shape_ = (160, 3754)
    else:
        shape_ = (len, 3750)

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

if __name__ == '__main__':

    data_ = import_OM("unlabeled")
    print(np.shape(data_))