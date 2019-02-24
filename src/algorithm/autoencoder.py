import torch
from torch.utils.data import DataLoader
from src.utils import constants
from src.legacy.TABaseline.code.ecgdataset import ECGDataset
from torch import nn, optim
from torch.nn import functional as F
from src.data.unlabelled_data import UnlabelledDataset
import configparser
import sys
from src.legacy.TABaseline.code import Preprocessor as pp

# TODO:
# -Done 1. use config parser
# -Done 2. create DataSet class for unlabelled data, or find under legacy


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.preprocess = pp.Preprocessor()

        # the number of hidden units are hardcoded for now.
        self.encoder = nn.Sequential(
            nn.Linear(3750, 1024), nn.ReLU(True), nn.Linear(1024, 256), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 1024), nn.ReLU(True), nn.Linear(1024, 3750)
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def plot_signal(output):
    print(output)
