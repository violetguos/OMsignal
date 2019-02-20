# this module augments the dataset through various methods

import numpy as np


def upside_down_inversion(x):
    # x = time series, dimension = n_values
    # negate the values of x
    x_flip = (-1.0 * x).copy()
    return x_flip


def adding_partial_noise(x, second=10, duration=2, num_samples_per_second=125):
    # x = time series, dimension = n_values
    # replace the signal belonging to interval [second, second+duration] by
    # a noise that matches the mean and variance of the dropped samples.
    begin = second * num_samples_per_second
    end = (second + duration) * num_samples_per_second
    section = x[begin:end]
    delta_x = np.std(section)
    mean_x = np.mean(section)
    noise = np.random.normal(mean_x, delta_x, end - begin)
    x_noise = x.copy()
    x_noise[begin:end] = noise
    return x_noise


def shift_series(x, shift=10):
    # shift the time series by shift steps
    shifted_x = np.concatenate((x[shift:], x[:shift] + x[-1] - x[0]))
    return shifted_x
