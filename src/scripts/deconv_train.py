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
import matplotlib.pyplot as plt


# BEGIN Global variables #
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

# fix the seed for reproducibility
seed = 54
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


targets = constants.TARGETS


# END global variables #


def layer_plot(x, title="ladder", fig="ladder"):
    """
    :param x: a dictionary of (key, numpy arrays) to plot
    :param fig: name of the figure
    :return:
    """
    plt.title(title)

    for key, val in x.items():
        plt.plot(
            val.data.cpu().numpy().reshape(constants.SHAPE_OF_ONE_DATA_POINT[1]),
            label=key,
        )

    plt.legend(loc="best")

    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()


def train_ae_mse_per_epoch(model, criterion, loader, plot=False):
    model.train()
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        data = model.preprocess_norm(data, batch_size=16)
        output_recon = model(data)
        mse_loss = criterion(output_recon, data)
    if plot:
        return mse_loss, data, output_recon
    return mse_loss



def train_prediction_per_epoch(model, criterion,loader):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        # only training on class label

        target = target[3]
        target = target.to(device)
        target = target.squeeze()

        data = model.preprocess_norm(data, batch_size=16)
        _, output_pred = model(data, prediction=True)
        ce_loss = criterion(output_pred, target)
    return ce_loss


def train_all(unlabelled_loader, train_loader, valid_loader, num_epoch):
    model = CnnDeconvAutoEncoder(1, 8)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion_recon = nn.MSELoss()
    criterion_pred = nn.CrossEntropyLoss()

    # TODO: add num epoch
    for i in range(num_epoch):
        optimizer.zero_grad()

        print("Training epoch {}".format(i))
        mse_loss, batch_data, batch_out = train_ae_mse_per_epoch(
            model, criterion_recon, unlabelled_loader, plot=True
        )
        ce_loss = train_prediction_per_epoch(model, criterion_pred, train_loader)

        loss = mse_loss + 10* ce_loss
        loss.backward()
        optimizer.step()

        epoch_plot = {"data": batch_data[0], "out": batch_out[0]}
        # layer_plot(epoch_plot, fig="real_data" + str(i))
        print("Evaluating epoch{}".format(i))
        evaluate_performance(model, valid_loader, i)


def load_data(batchsize):
    print("Load data")
    # TODO: read model dict, hardcode for now
    train_dataset = ecgdataset.ECGDataset(
        constants.T5_FAKE_VALID_LABELED_DATA, True, target=targets
    )
    valid_dataset = ecgdataset.ECGDataset(
        constants.T5_FAKE_VALID_LABELED_DATA, False, target=targets
    )

    unlabeled_dataset = UnlabelledDataset(constants.T5_FAKE_VALID_LABELED_DATA, False)

    train_loader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batchsize, shuffle=False, num_workers=1)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batchsize, shuffle=True, num_workers=1, drop_last=True
    )
    print("Finished data loading")

    return train_loader, valid_loader, unlabeled_loader


def evaluate_performance(model, valid_loader, e):
    correct = 0.0
    total = 0.0
    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        # a hack for now
        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        if use_gpu:
            data = data.cuda()
        target = target[3] #only read the IDs
        target = target.to(device=device)
        target = target.squeeze()
        data = model.preprocess_norm(data, batch_size=16)
        # print("eval data", data.shape)

        _, output = model.forward(data, prediction=True)
        # print("target", target)
        if use_gpu:
            output = output.cpu()
            target = target.cpu()

        output = output.detach().numpy()
        target = target.data.numpy()

        preds = np.argmax(output, axis=1)
        # print("preds", preds)
        correct += np.sum(target == preds)
        total += target.shape[0]

    print("Epoch:", e + 1, "\t", "Validation Accuracy:", correct / total)


def main():
    # TODO: add the config script
    num_epoch = 50
    batchsize = 16
    print("batchsize", batchsize)
    print("hello i'm training")
    train_loader, valid_loader, unlabeled_loader = load_data(batchsize)
    train_all(unlabeled_loader,train_loader, valid_loader, num_epoch)


if __name__ == "__main__":
    main()
