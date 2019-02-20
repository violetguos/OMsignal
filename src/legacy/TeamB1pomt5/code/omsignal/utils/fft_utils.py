
"""
This file is not needed anymore, keeping just in case for now
"""

import numpy as np
import torch

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


def plot_ecgfft(x, y):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    # plots the real and imaginary part of the FFT of an ECG signal
    plt.title('ECG FFT')
    plt.plot(x[0, 0, :])
    plt.plot(y[0, 0, :])
    plt.xlabel('Frequency')
    plt.ylabel('FFT')
    plt.legend(['Real', 'Imag'])
    plt.savefig('fft_visual.png')
    plt.close()

if __name__ == '__main__':
    # only for testing; this reads and create a plot of an FFT
    fake_ecg = np.random.randn(3750).astype(np.float32)
    fftr, ffti = make_fft(fake_ecg)
    print(fftr.shape)
    #plot_ecgfft(fftr, ffti)