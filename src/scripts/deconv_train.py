from src.algorithm.cnn_deconv_autoenoder import CnnDeconvAutoEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from src.data.unlabelled_data import UnlabelledDataset
import src.legacy.TABaseline.code.ecgdataset as ecgdataset
from src.utils import constants

# BEGIN Global variables #
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

# fix the seed for reproducibility
seed = 54
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

target_out_size_dict = {"pr_mean": 1, "rt_mean": 1, "rr_stdev": 1, "userid": 32}
target_criterion_dict = {
    "pr_mean": nn.MSELoss(),
    "rt_mean": nn.MSELoss(),
    "rr_stdev": nn.MSELoss(),
    "userid": nn.CrossEntropyLoss(),
}

targets = constants.TARGETS

# Hyperparameter ratio for unsupervised propagation
LR_RATIO = 0.1
BATCHSIZE_RATIO = 2

# END global variables #


def train_deconv(model, optimizer, criterion, loader):

    model.train()
    for batch_idx, (data, _) in enumerate(loader):
        data = model.preprocess_norm(data, batch_size=16)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
    return loss


def train_epoch(loader):
    model = CnnDeconvAutoEncoder(1, 8)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion_recon = nn.MSELoss()
    train_deconv(model, optimizer, criterion_recon, loader)

    for i in range(10):
        loss = train_deconv(model, optimizer, criterion_recon, loader)
        print(loss)



def load_data(batchsize):
    # TODO: read model dict, hardcode for now
    train_dataset = ecgdataset.ECGDataset(
        constants.T5_FAKE_TRAIN_LABELED_DATA, True, target=targets
    )
    valid_dataset = ecgdataset.ECGDataset(
        constants.T5_FAKE_TRAIN_LABELED_DATA, False, target=targets
    )

    unlabeled_dataset = UnlabelledDataset(constants.T5_FAKE_TRAIN_LABELED_DATA, False)

    train_loader = DataLoader(
        train_dataset,batchsize, shuffle=True, num_workers=1
    )
    valid_loader = DataLoader(
        valid_dataset, batchsize, shuffle=False, num_workers=1
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batchsize,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    return train_loader, valid_loader, unlabeled_loader


def main():
    batchsize = 16
    train_loader, valid_loader, unlabeled_loader = load_data(batchsize)
    train_epoch(unlabeled_loader)


if __name__ == "__main__":
    main()