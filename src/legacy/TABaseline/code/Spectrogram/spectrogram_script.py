# WIP
# Simon Blackburn 2018-12-17

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import Preprocessor
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 2

FS = 64
NPERSEG = 1000
WINDOW = 'hann'


def read_data():
    # read binary data and return as a numpy array
    fp = np.memmap('../MILA_TrainLabeledData.dat',
                   dtype='float32', mode='r', shape=(160, 3754))
    trainData = np.zeros(shape=(160, 3754))
    trainData[:] = fp[:]
    del fp
    return torch.Tensor(trainData[:, :3750])


def plot_spectrogram(x, lognorm=False):
    f, t, Zxx = signal.spectrogram(x, fs=64, nperseg=16, noverlap=16//4)
    if lognorm:
        Zxx = np.abs(Zxx)
        mask = Zxx > 0
        Zxx[mask] = np.log(Zxx[mask])
        Zxx = (Zxx - np.min(Zxx)) / (np.max(Zxx) - np.min(Zxx))
    plt.title('Spectrogram')
    plt.pcolormesh(t, f, Zxx, vmin=0, vmax=1)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig('stft.png')
    plt.close()


if __name__ == '__main__':
    series = read_data()

    ecg = read_data()
    ecg = ecg.unsqueeze(1)
    ecg_loader = DataLoader(ecg, batch_size=BATCH_SIZE, shuffle=False)
    model = Preprocessor.Preprocessor()
    for i, x in enumerate(ecg_loader):
        y = model(x)
        if i == 0:
            plot_spectrogram(y[0][0].numpy(), True)
            break

    #fourier_series = np.zeros((series.shape[0],series.shape[1]//2+1))
    # for k in range(len(series)):
    #    fourier_series[k] = np.fft.rfft(series[k])
    print('ok')
