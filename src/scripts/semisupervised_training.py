import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from tensorboardX import SummaryWriter

import src.legacy.TABaseline.code.baseline_models as models
import src.legacy.TABaseline.code.scoring_function as scoreF
import src.legacy.TABaseline.code.ecgdataset as ecgdataset
from src.algorithm.autoencoder import AutoEncoder, CnnAutoEncoder
from src.legacy.TABaseline.code import Preprocessor as pp
from src.data.unlabelled_data import UnlabelledDataset
from src.legacy.TABaseline.code.baseline_multitask_main import (
    eval_model,
    train_model,
    training_loop,
    load_model,
)
from src.algorithm.CNN_multitask_semisupervised import Conv1DBNLinear
from src.utils import constants
from src.utils.cache import ModelCache
from src.utils.os_helper import get_hyperparameters
import sys
import configparser
import os


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
LR_RATIO = 1
BATCHSIZE_RATIO = 4

# END global variables #


def save_model(epoch, model, prefix='neural_network', path='./'):

    # creating a filename indexed by the epoch value
    filename = path + prefix + '_{}.pt'.format(epoch)

    # save the parameters of the model.
    torch.save(model.state_dict(), filename)


def train_unsupervised_per_epoch(model, optimizer, batch_size, unlabeled_loader):
    """
    Trains model when unlabeled data is given
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param batch_size: size of one mini-batch
    :param unlabeled_loader: the unlabelled data set loader, not the TRAIN_LABELLED_DATA
    :return: at the end of epoch, the MSE loss
    """
    model.train()

    for batch_idx, (data, _) in enumerate(unlabeled_loader):
        data = data.to(device)
        # accommodates the TA's preprocessor dimension
        data = data.view(batch_size, 1, 3750)

        optimizer.zero_grad()
        output = model(data, label=False)

        data = pp.Preprocessor().forward(data)
        MSE_loss = nn.MSELoss()(output, data)

        # ===================backward====================
        optimizer.zero_grad()
        MSE_loss.backward()
        optimizer.step()

    return MSE_loss.item()


def training_loop(
        model,
        optimizer_encoder,
        optimizer_prediction,
        criterion,
        train_loader,
        unlabeled_loader,
        eval_loader,
        score_param_index,
        hyperparameters_dict,
        loss_history,
        chkptg_freq=10,
        prefix='neural_network',
        path='./'):
    # train the model using optimizer / criterion
    # this function also creates a tensorboard log
    writer = SummaryWriter(hyperparameters_dict['tbpath'])

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    weight = hyperparameters_dict['weight']

    loss_history.prefix = "CNNencoder"
    loss_history.mode = "train"
    for epoch in range(1, hyperparameters_dict['nepoch'] + 1):

        train_mse_loss = train_unsupervised_per_epoch(
            model,
            optimizer_encoder,
            hyperparameters_dict["batchsize"]*BATCHSIZE_RATIO,
            unlabeled_loader,
        )
        # log the errors everytime!
        loss_history.log(epoch, train_mse_loss)

        train_loss, train_acc = train_model(
            model, optimizer_prediction, criterion, train_loader,
            score_param_index, weight
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        valid_loss, valid_acc = eval_model(
            model, criterion, eval_loader,
            score_param_index, weight
        )
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)

        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Valid/Loss', valid_loss, epoch)

        writer.add_scalar('Training/OverallScore', train_acc[0], epoch)
        writer.add_scalar('Valid/OverallScore', valid_acc[0], epoch)

        writer.add_scalar('Training/prMeanTau', train_acc[1], epoch)
        writer.add_scalar('Valid/prMeanTau', valid_acc[1], epoch)

        writer.add_scalar('Training/rtMeanTau', train_acc[2], epoch)
        writer.add_scalar('Valid/rtMeanTau', valid_acc[2], epoch)

        writer.add_scalar('Training/rrStdDevTau', train_acc[3], epoch)
        writer.add_scalar('Valid/rrStdDevTau', valid_acc[3], epoch)

        writer.add_scalar('Training/userIdAcc', train_acc[4], epoch)
        writer.add_scalar('Valid/userIdAcc', valid_acc[4], epoch)

        print("Epoch {} {} {} {} {}".format(
            epoch, train_loss, valid_loss, train_acc, valid_acc)
        )

        # Checkpoint
        if epoch % chkptg_freq == 0:
            save_model(epoch, model, prefix, path)

    save_model(hyperparameters_dict['nepoch'], model, prefix, path)
    return [
        (train_loss_history, train_acc_history),
        (valid_loss_history, valid_acc_history)
    ]


def trainer_prediction(model_hp_dict,
                       train_loader,
                       unlabeled_loader,
                       valid_loader,
                       loss_history):

    # Model initialization
    target_labels = targets.split(",")
    target_labels = [s.lower().strip() for s in target_labels]
    if len(target_labels) == 1:
        out_size = target_out_size_dict[target_labels[0]]
    else:
        out_size = [target_out_size_dict[a] for a in target_labels]
    n_layers = model_hp_dict["n_layers"]
    hidden_size = model_hp_dict["hidden_size"]
    kernel_size = model_hp_dict["kernel_size"]
    pool_size = model_hp_dict["pool_size"]
    dropout = model_hp_dict["dropout"]

    # Changed from TA's legacy code file path
    # constants.SAVE_MODEL_PATH + "/" # hack for TA's legacy code, not my fault

    path = loss_history.dir + "/"

    loss_history.prefix = "cnn_model_encoder_part_epoch"
    model_hp_dict["tbpath"] = os.path.join(path, "tensorboard")
    chkptg_freq = model_hp_dict["nepoch"] // 10

    # define the model
    model = Conv1DBNLinear(
        1, out_size, hidden_size, kernel_size, pool_size, dropout
    )
    # model to gpu, create optimizer, criterion and train
    model.to(device)
    optimizer_prediction = optim.Adam(model.parameters(),
                                      lr=model_hp_dict['learning_rate'])
    optimizer_encoder = optim.Adam(model.parameters(),
                                      lr=model_hp_dict['learning_rate']*LR_RATIO)

    if len(target_labels) == 1:
        criterion = target_criterion_dict[target_labels[0]]
    else:
        criterion = [target_criterion_dict[a] for a in target_labels]

    scoring_func_param_index = [
        None if target_labels.count("pr_mean") == 0 else target_labels.index("pr_mean"),
        None if target_labels.count("rt_mean") == 0 else target_labels.index("rt_mean"),
        None
        if target_labels.count("rr_stdev") == 0
        else target_labels.index("rr_stdev"),
        None if target_labels.count("userid") == 0 else target_labels.index("userid"),
    ]

    # Training loop for CNN from ta BASELINE model
    train, valid = training_loop(
        model,
        optimizer_encoder,
        optimizer_prediction,
        criterion,
        train_loader,
        unlabeled_loader,
        valid_loader,
        scoring_func_param_index,
        model_hp_dict,
        loss_history,
        chkptg_freq,
        loss_history.prefix,
        path,
    )

    print("save cnn model", loss_history.dir)
    np.savetxt(os.path.join(loss_history.dir, "cnn_train.txt"), np.array(train[0]))
    np.savetxt(os.path.join(loss_history.dir, "cnn_valid.txt"), np.array(valid[0]))

    print("CNN training Done")


def load_data(model_hp_dict):

    train_dataset = ecgdataset.ECGDataset(
        constants.TRAIN_LABELED_DATASET_PATH, True, target=targets
    )
    valid_dataset = ecgdataset.ECGDataset(
        constants.VALID_LABELED_DATASET_PATH, False, target=targets
    )

    unlabeled_dataset = UnlabelledDataset(constants.UNLABELED_DATASET_PATH, False)

    train_loader = DataLoader(
        train_dataset, model_hp_dict["batchsize"], shuffle=True, num_workers=1
    )
    valid_loader = DataLoader(
        valid_dataset, model_hp_dict["batchsize"], shuffle=False, num_workers=1
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        model_hp_dict["batchsize"]*BATCHSIZE_RATIO,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )
    return train_loader, valid_loader, unlabeled_loader


def run(model_hp_dict):
    loss_history = ModelCache(_prefix="unlabeled_pass", _mode="train")

    train_loader, valid_loader, unlabeled_loader = load_data(
        model_hp_dict
    )

    # then append the training on CNN from block 1
    trainer_prediction(
        model_hp_dict, train_loader, unlabeled_loader, valid_loader, loss_history
    )



def main(config_model):

    # reads in the config files
    model_config = configparser.ConfigParser()
    model_config.read(config_model)
    model_hp_dict = get_hyperparameters(model_config)

    # Call functions to run models
    run(model_hp_dict)


if __name__ == "__main__":

    # Read the ini file name from sys arg to avoid different people's different local set up
    # Use a shell script instead to run on your setup

    #main(sys.argv[1], sys.argv[2])
    main("src/scripts/model_input.in")

