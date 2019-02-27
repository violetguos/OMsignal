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
