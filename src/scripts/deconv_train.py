from src.algorithm.cnn_deconv_autoenoder import CnnDeconvAutoEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from src.data.unlabelled_data import UnlabelledDataset
import src.legacy.TABaseline.code.ecgdataset as ecgdataset
from src.utils import constants
import matplotlib.pyplot as plt
from src.utils.cache import ModelCache
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor




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

def noise(noise_level, data):
    noise = np.random.normal(
        loc=0.0, scale=noise_level, size=data.size()
    )
    if use_gpu:
        noise = Variable(torch.cuda.FloatTensor(noise))
    else:
        noise = Variable(torch.FloatTensor(noise))
    return noise


def layer_plot(x, title="ladder", fig="ladder"):
    """
    :param x: a dictionary of (key, numpy arrays) to plot
    :param fig: name of the figure
    :return:
    """
    plt.title(title)

    for key, val in x.items():
        plt.plot(
            val.data.cpu().numpy().reshape(constants.FFT_SHAPE[1]),
            label=key,
        )

    plt.legend(loc="best")

    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()


def train_ae_mse_per_epoch(model, criterion, loader, batch_size, plot=False):
    model.train()
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        # preprocess = Preprocessor()
        # preprocess.to(device)
        data = torch.squeeze(data)
        # print("data", data.size())

        # data = preprocess(data)
        # print("data", data.shape)

        data = model.preprocess_norm(data, batch_size=batch_size)
        output_recon = model(data)
        mse_loss = criterion(output_recon, data)
    if plot:
        data = data.to(device)
        data = model.preprocess_norm(data, batch_size=batch_size)
        return mse_loss, data, output_recon
    return mse_loss



def train_prediction_per_epoch(model, criterion,loader, batch_size):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        # preprocess = Preprocessor()
        # preprocess.to(device)
        data = torch.squeeze(data)

        # data = preprocess(data)

        # only training on class label
        # print("data", data.shape)

        target = target[3]
        target = target.to(device)
        target = target.squeeze()

        data = model.preprocess_norm(data, batch_size=batch_size)
        data += noise(0.05, data)
        _, output_pred = model(data, prediction=True)
        ce_loss = criterion(output_pred, target)
    train_acc = classification_accuracy(output_pred, target)
    return ce_loss, train_acc


def classification_accuracy(output, target):
    correct = 0.0
    total = 0.0
    if use_gpu:
        output = output.cpu()
        target = target.cpu()
    output = output.detach().numpy()
    target = target.data.numpy()
    # print("output", output.shape)
    preds = np.argmax(output, axis=1)
    # print("preds", preds)
    correct += np.sum(target == preds)
    total += target.shape[0]
    accuracy = correct/total
    return accuracy



def train_all(unlabelled_loader, train_loader, valid_loader, num_epoch, batch_size):
    cache = ModelCache()
    model = CnnDeconvAutoEncoder(1, 8)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion_recon = nn.MSELoss()
    criterion_pred = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau()


    # TODO: add num epoch
    model_save_freq = 9
    for i in range(num_epoch):
        optimizer.zero_grad()

        print("Training epoch {}".format(i))
        mse_loss, batch_data, batch_out = train_ae_mse_per_epoch(
            model, criterion_recon, unlabelled_loader, batch_size, plot=True
        )

        ce_loss, train_acc = train_prediction_per_epoch(model, criterion_pred, train_loader, batch_size)
        # train_prediction_per_epoch(model, criterion_pred, train_loader, batch_size)

        loss = 10 * mse_loss + ce_loss
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)


        print("Evaluating epoch{}".format(i))
        print("Epoch {} \t train accuracy {}".format(i+1, train_acc))
        print("Epoch {} \t train MSE loss {}".format(i+1, mse_loss))

        valid_acc, valid_loss = evaluate_performance(model, valid_loader, i, batch_size)
        cache.scalar_summary('Train_acc', train_acc, i)
        cache.scalar_summary('Train_cross_entropy', ce_loss, i)
        cache.scalar_summary('Valid_acc', valid_acc, i)
        cache.scalar_summary('Valid_cross_entropy', valid_loss, i)
        cache.scalar_summary('train_mse_loss', mse_loss, i)
        if i % model_save_freq:
            cache.save(model, i)
            epoch_plot = {"data": batch_data[0], "out": batch_out[0]}
            layer_plot(epoch_plot, fig="real_data" + str(i))
    cache.save(model, num_epoch)


def load_data(batchsize):
    print("Load data")
    train_dataset = ecgdataset.ECGDataset(
        constants.TRAIN_LABELED_DATASET_PATH, use_transform=True, target=targets, use_fft=True
    )

    valid_dataset = ecgdataset.ECGDataset(
        constants.VALID_LABELED_DATASET_PATH, use_transform=False, target=targets, use_fft=True
    )

    unlabeled_dataset = UnlabelledDataset(constants.UNLABELED_DATASET_PATH, use_transform=True)

    train_loader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batchsize, shuffle=False, num_workers=1)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batchsize, shuffle=True, num_workers=1, drop_last=True
    )
    print("Finished data loading")

    return train_loader, valid_loader, unlabeled_loader


def evaluate_performance(model, valid_loader, e, batch_size):
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
        data = model.preprocess_norm(data, batch_size=batch_size)

        _, output = model.forward(data, prediction=True)
        if use_gpu:
            output = output.cpu()
            target = target.cpu()

        output = output.detach()
        valid_loss = nn.CrossEntropyLoss()(output, target)

        output = output.numpy()
        target = target.data.numpy()

        preds = np.argmax(output, axis=1)
        correct += np.sum(target == preds)
        total += target.shape[0]

    valid_acc = correct / total
    print("Epoch:", e + 1, "\t", "Validation Accuracy:", valid_acc, "Validation loss: ", valid_loss.item() )
    return valid_acc, valid_loss.item()


def main():
    # TODO: add the config script
    num_epoch = 50
    batchsize = 64
    print("batchsize", batchsize)
    print("hello i'm training")
    train_loader, valid_loader, unlabeled_loader = load_data(batchsize)
    train_all(unlabeled_loader,train_loader, valid_loader, num_epoch, batchsize)


if __name__ == "__main__":
    main()
