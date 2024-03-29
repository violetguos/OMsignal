import os


"""
Contains all os related helper functions e.g.
File size
Reading config files, jsons
"""


def get_num_data_points(file_path, size_of_one_data_point_bytes):
    # calculates number of data points in a memmap file, works for any file
    file_size_bytes = os.path.getsize(file_path)
    return int(file_size_bytes / size_of_one_data_point_bytes)


def get_hyperparameters(config, autoencoder=False):
    """
    :param config: an .in file with params for a model
    :param autoencoder: Boolean, to include an autoencoder in CNN or not
    :return: a hyperparam dictionary
    """
    hyperparam = {}
    hyperparam["learning_rate"] = float(config.get("optimizer", "learning_rate"))
    hyperparam["momentum"] = float(config.get("optimizer", "momentum"))
    hyperparam["batchsize"] = int(config.get("optimizer", "batch_size"))
    hyperparam["nepoch"] = int(config.get("optimizer", "nepoch"))
    hyperparam["model"] = config.get("model", "name")
    hyperparam["hidden_size"] = int(config.get("model", "hidden_size"))
    hyperparam["dropout"] = float(config.get("model", "dropout"))
    hyperparam["n_layers"] = int(config.get("model", "n_layers"))
    if not autoencoder:
        hyperparam["kernel_size"] = int(config.get("model", "kernel_size"))
        hyperparam["pool_size"] = int(config.get("model", "pool_size"))
        hyperparam["tbpath"] = config.get("path", "tensorboard")
        hyperparam["modelpath"] = config.get("path", "model")
        weight1 = float(config.get("loss", "weight1"))
        weight2 = float(config.get("loss", "weight2"))
        weight3 = float(config.get("loss", "weight3"))
        weight4 = float(config.get("loss", "weight4"))
        hyperparam["weight"] = [weight1, weight2, weight3, weight4]
    return hyperparam


#!/usr/bin/env python3
'''
Module for reading/writing numpy memfile
'''

import numpy as np


def read_memfile(filename: str, shape: tuple, dtype: str = 'float32') -> np.ndarray:
    """Read binary data and return as a numpy array of desired shape

    Args:
        filename: Path of memfile.
        shape: Shape of numpy array.
        dtype (:obj:`str`, optional): numpy dtype. Defaults to ``float32``.

    Returns:
        ndarray: A numpy ndarray with data from memfile.
    """
    # read binary data and return as a numpy array
    fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    data = np.zeros(shape=shape, dtype=dtype)
    data[:] = fp[:]
    del fp
    return data


def write_memfile(data: np.ndarray, filename: str) -> None:
    """Writes ``data`` to file specified by ``filename``.

    Args:
        data: ndarray containing the data.
        filename: Name of memfile to be created with contents of ``data``.

    Returns:
        None
    """
    shape = data.shape
    dtype = data.dtype
    fp = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    fp[:] = data[:]
    del fp
