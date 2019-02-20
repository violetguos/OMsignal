import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import Preprocessor
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 2


def read_data():
    # read binary data and return as a numpy array
    fp = np.memmap('MILA_TrainLabeledData.dat',
                   dtype='float32', mode='r', shape=(160, 3754))
    trainData = np.zeros(shape=(160, 3754))
    trainData[:] = fp[:]
    del fp
    return (
        torch.Tensor(trainData[:, :3750]),
        torch.Tensor(trainData[:, 3750]),
        torch.Tensor(trainData[:, 3751]),
        torch.Tensor(trainData[:, 3752]),
        torch.Tensor(trainData[:, 3753]).type(torch.int32)
    )


def plot_ecg(x, y):
    # simple plot function, before and after transform
    plt.title('Before preprocessor')
    for b in range(BATCH_SIZE):
        plt.plot(x[b, 0, :])
    plt.xlabel('time step (30 sec * 125 Hz)')
    plt.ylabel('ECG measure')
    plt.savefig('ecg_beforepp.png')
    plt.close()
    plt.title('After preprocessor')
    for b in range(BATCH_SIZE):
        plt.plot(y[b, 0, :])
    plt.xlabel('time step (30 sec * 125 Hz)')
    plt.ylabel('ECG measure')
    plt.savefig('ecg_afterpp2.png')
    plt.close()


if __name__ == '__main__':
    ecg, pr_mean, rt_mean, rr_stdev, userid = read_data()
    ecg = ecg.unsqueeze(1)
    ecg_loader = DataLoader(ecg, batch_size=BATCH_SIZE, shuffle=False)
    model = Preprocessor.Preprocessor()
    for i, x in enumerate(ecg_loader):
        y = model(x)
        if i == 0:
            print(x.size(), y.size())
            plot_ecg(x.numpy(), y.numpy())
