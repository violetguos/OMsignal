import sys
import os
import torch
from src.utils import constants
from src.legacy.TABaseline.code.ecgdataset import ECGDataset
from src.algorithm.autoencoder import AutoEncoder
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

from src.data.unlabelled_data import UnlabelledDataset
from src.legacy.TABaseline.code import Preprocessor as pp


'''creates a plot to compare before and after basic autoencoder reconstruction of a signal'''


def plot_signal(x, title="Signal MSE=0.4", fig="ae_signal"):
    """
    :param x: a dictionary of (key, numpy arrays) to plot
    :param fig: name of the figure
    :return:
    """
    plt.title(title)

    for key, val in x.items():
        plt.plot(val.data.cpu().numpy().reshape(constants.SHAPE_OF_ONE_DATA_POINT[1]), label=key)


    plt.legend(loc='best')

    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()

def run(model, data):
    model.eval()
    out = model(data)
    return out

def read():
    unlabeled_dataset = UnlabelledDataset(constants.UNLABELED_DATASET_PATH, False)
    out = pp.Preprocessor().forward(unlabeled_dataset[0][0])
    print(out.shape)
    return out


def load_model(args):
    model = AutoEncoder()
    device = args.device
    state_dict = torch.load(args.model_path, map_location=args.device)

    model.load_state_dict(state_dict)
    model.to(device)

    return model


def main(argv):
    parser = argparse.ArgumentParser(description='plot signals')
    parser.add_argument('--model_path', type=str, default='', help='model pt directory')

    args = parser.parse_args(argv)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    args.cuda = use_gpu
    args.device = device

    # model load
    model = load_model(args)

    # read from the dataset
    x_original = read()
    x_ae = run(model, x_original)

    plot_var = {'original signal':x_original, 'autodencoder reconstruct': x_ae}
    plot_signal(plot_var)


def extract_model_time(args):
    """
    :param args: argument, particularly looks for the YYYY_MM_DD_HH_MM_SS
    :return: the substring in YYYY_MM_DD_HH_MM_SS format
    """
    path_list = args.model_path.split(os.sep)
    return path_list[3]


if __name__ == "__main__":
    main(['--model_path=../../log/2019_02_26_14_42_31_/model/autoencoder_epoch_12.pt'])
