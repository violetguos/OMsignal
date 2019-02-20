'''
Functions for visualization of data
as simple line graphs and FFT.
'''

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
import sys
sys.path.append(os.path.join('..', '..'))   # Hack - fix package management later

from omsignal.utils.dataloader_utils import import_OM
from omsignal.utils.fft_utils import make_fft

def plot_lines(samples, series, window, output_path, freq=125):
    title = 'Sample {}'.format(samples)
    plt.subplot(len(samples), 1, 1)
    plt.title(title)
    plt.plot(np.divide(window, freq), series[0][window])
    for i in range(len(samples)-1):
        plt.subplot(len(samples), 1, i+2)
        plt.plot(np.divide(window, freq), series[i+1][window])
    plt.savefig(output_path)
    plt.close()


def normalize_samples(train_data, samples, series, nb_datapoints=3750):
    for i, sample in enumerate(samples):
        if np.max(train_data[sample]) > -np.min(train_data[sample]):
            series[i] = np.divide(train_data[sample, range(nb_datapoints)], \
                np.max(train_data[sample, range(nb_datapoints)]))
        else:
            series[i] = np.divide(train_data[sample, range(nb_datapoints)], \
                -np.min(train_data[sample, range(nb_datapoints)]))
    return series

def plot_ecgfft(samples, fft_series, output_path):
    print(len(samples))
    # plots the real part of the FFT of an ECG signal
    plt.subplot(len(samples), 1, 1)
    plt.title('ECG FFT - Samples {}'.format(samples))
    for i in range(len(samples)):
        plt.subplot(len(samples), 1, i+1)   
        plt.plot(fft_series[i])
        plt.xlim(1, 1750)
        #plt.ylim(-1, 1)    # This is bugged for some reason
    plt.xlabel('Frequency')
    plt.ylabel('FFT')
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Generate line graph visualizations of the data.')
    parser.add_argument('nb_plots', help='Number of samples to plot.', type=int)
    parser.add_argument('--cluster', help='Flag for running on the Helios cluster. \
        If flagged, will use real data; otherwise, will use dummy data.', action='store_true')
    parser.add_argument('--normalize', help='Normalize samples.', action='store_true')
    parser.add_argument('--truncate', nargs=2, help='Truncate samples to a specific \
        time interval (in seconds, between 0 and 30 inclusive). Example use: --truncate 1 2')
    parser.add_argument('--norandom', nargs='+', help='Specify sample indices \
        (instead of random default). Example use: --norandom 15 30 66')
    args = parser.parse_args()

    # Magic numbers - always true for this dataset
    freq = 125              # Sampling frequency in Hz
    nb_datapoints = 3750    # 30 seconds x 125 Hz = 3750

    # Set random seeds for reproducibility
    np.random.seed(43)
    random.seed(43)

    # Check truncation arguments
    if args.truncate:
        start_second = int(args.truncate[0])        # Ensure casted as ints
        end_second = int(args.truncate[1])
        for second in [start_second, end_second]:
            if second < 0 or second > 30:
                raise ValueError('Interval markers (in seconds) must be between 0 and 30 inclusive.')
        if end_second <= start_second:
            raise ValueError('Interval end must be greater than interval beginning.')
        # Create plot window according to the initial parameters
        window = range(start_second*freq, end_second*freq)
    else:
        window = range(nb_datapoints)

    # Load real vs dummy data if we are on cluster vs local, respectively
    ecg, _, _, _, _ = import_OM('train', cluster=args.cluster)
    train_data = ecg
    train_data = np.squeeze(train_data, axis=(1,))
    nb_data = len(train_data)

    # Check sampling arguments
    if args.norandom:
        samples = [int(x) for x in args.norandom]   # Ensure casted as ints
        if len(samples) != args.nb_plots:
            raise ValueError('Number of sample indices not equal to number of plots.')
    else:
        samples = np.sort(random.sample(range(nb_data), args.nb_plots))

    # Normalize if necessary
    series = np.zeros((args.nb_plots, nb_datapoints))
    if args.normalize:
        series = normalize_samples(train_data, samples, series, nb_datapoints=nb_datapoints)
    else:
        for i, sample in enumerate(samples):
            series[i] = train_data[sample, range(nb_datapoints)]

    # Make the line plots
    fig_name = 'train_{}'.format('_'.join([str(x) for x in samples]))
    if args.normalize:
        fig_name += '_norm'
    if args.truncate:
        fig_name += '_trunc_{}_{}'.format(start_second, end_second)
    fig_name += '.png'
    output_path = os.path.join('..', '..', 'figures', fig_name)
    plot_lines(samples, series, window, output_path, freq=125)

    # FFT
    fig_name = fig_name.replace('.png', '')
    fig_name += '_fft.png'
    output_path = os.path.join('..', '..', 'figures', fig_name)
    fft_series = [make_fft(x) for x in series]
    plot_ecgfft(samples, fft_series, output_path)
    '''for i, sample in enumerate(samples):
        fig_name = fig_name.replace('.png', '')
        fig_name += '_fft.png'
        output_path = os.path.join('..', '..', 'figures', fig_name)
        fft_r, fft_i = make_fft(series[i])
        plot_ecgfft(fft_r, fft_i, sample, output_path)'''