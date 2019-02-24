import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# from tensorboardX import SummaryWriter

import src.legacy.TABaseline.code.baseline_models as models
import src.legacy.TABaseline.code.scoring_function as scoreF
import src.legacy.TABaseline.code.ecgdataset as ecgdataset
from src.algorithm.autoencoder import AutoEncoder
from src.legacy.TABaseline.code import Preprocessor as pp
from src.data.unlabelled_data import UnlabelledDataset
from src.legacy.TABaseline.code.baseline_multitask_main import (
    eval_model,
    train_model,
    training_loop,
    load_model,
)
from src.algorithm.CNN_multitask import Conv1DBNLinear
from src.utils import constants
from src.utils.cache import ModelCache
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
# END global variables #


# TODO: move to the util/os_helper.py
def get_hyperparameters(config, autoencoder=False):
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


def train_autoencoder_per_epoch(epoch, model, optimizer, batch_size, train_loader):
    """
    Trains the autoeoncoder on one epoch, called by an outer loop that controls total number of epoches
    :param epoch: current number of epoch
    :param model: the autoencoder model created under src/algorithms
    :param optimizer: pytorch optim
    :param batch_size: size of one mini-batch
    :param train_loader: the unlabelled data set loader, not the TRAIN_LABELLED_DATA
    :return: at the end of epoch, the MSE loss
    """
    model.train()
    total_batch = constants.UNLABELED_SHAPE[0] // batch_size

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # accommodates the TA's preprocessor dimension
        data = data.view(batch_size, 1, 3750)

        optimizer.zero_grad()
        output = model(data)

        data = pp.Preprocessor().forward(data)
        MSE_loss = nn.MSELoss()(output, data)

        # ===================backward====================
        optimizer.zero_grad()
        MSE_loss.backward()
        optimizer.step()
        print(
            "epoch [{}], batch [{}/{}], MSE_loss:{:.4f}".format(
                epoch, batch_idx, total_batch, MSE_loss.data
            )
        )
    return MSE_loss.item()


def trainer_ae(autoencoder_hp_dict, unlabeled_loader, loss_history):
    # Autoencoder training
    save_freq = autoencoder_hp_dict["nepoch"] // 5
    autoencoder = AutoEncoder().to(device)
    AE_optimizer = optim.Adam(
        autoencoder.parameters(), lr=autoencoder_hp_dict["learning_rate"]
    )
    # TODO: make ModelCache for both the autoencoder part and the CNN part
    loss_history.prefix = "autoencoder"
    loss_history.mode = "train"
    for epoch in range(autoencoder_hp_dict["nepoch"]):
        train_mse_loss = train_autoencoder_per_epoch(
            epoch,
            autoencoder,
            AE_optimizer,
            autoencoder_hp_dict["batchsize"],
            unlabeled_loader,
        )
        # log the errors everytime!
        loss_history.log(epoch, train_mse_loss)

        if epoch % save_freq == 0:
            # Save model every save_freq epochs
            loss_history.save(autoencoder, epoch, verbose=True)

    print("Autoencoder training done")

    return autoencoder


def trainer_prediction(
    model_hp_dict, autoencoder, train_loader, valid_loader, loss_history
):

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

    loss_history.prefix = "autoencoder_cnn_model_cnn_part_epoch"
    model_hp_dict['tbpath'] = os.path.join(path, "tensorboard")
    chkptg_freq = model_hp_dict["nepoch"] // 10

    # define the model
    model = Conv1DBNLinear(
        1, out_size, hidden_size, kernel_size, pool_size, dropout, autoencoder=True
    )
    # model to gpu, create optimizer, criterion and train
    model.to(device)
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 0},
            {"params": model.decoder.parameters(), "lr": 0},
            {"params": model.batch_norm0.parameters()},
            {"params": model.batch_norm1.parameters()},
            {"params": model.batch_norm2.parameters()},
            {"params": model.batch_norm3.parameters()},
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters()},
            {"params": model.conv3.parameters()},
            {"params": model.conv4.parameters()},
            {"params": model.conv5.parameters()},
            {"params": model.conv6.parameters()},
            {"params": model.out.parameters()},
            {"params": model.nl.parameters()},
        ],
        lr=model_hp_dict["learning_rate"],
    )

    # only the encoder present in the prediction step
    model.encoder.load_state_dict(autoencoder.encoder.state_dict())

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
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        scoring_func_param_index,
        model_hp_dict,
        chkptg_freq,
        loss_history.prefix,
        path,
    )

    print("CNN training Done")


def load_data(model_hp_dict, autoencoder_hp_dict):

    train_dataset = ecgdataset.ECGDataset(
        constants.TRAIN_LABELED_DATASET_PATH, True, target=targets
    )
    valid_dataset = ecgdataset.ECGDataset(
        constants.VALID_LABELED_DATASET_PATH, False, target=targets
    )

    # testing on the small subset
    unlabeled_dataset = UnlabelledDataset(constants.UNLABELED_DATASET_PATH, False)

    train_loader = DataLoader(
        train_dataset, model_hp_dict["batchsize"], shuffle=True, num_workers=1
    )
    valid_loader = DataLoader(
        valid_dataset, model_hp_dict["batchsize"], shuffle=False, num_workers=1
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        autoencoder_hp_dict["batchsize"],
        shuffle=False,
        num_workers=1,
    )
    return train_loader, valid_loader, unlabeled_loader


def run(autoencoder_hp_dict, model_hp_dict):
    autoencoder_loss_history = ModelCache(_prefix="autoencoder", _mode="train")

    train_loader, valid_loader, unlabeled_loader = load_data(
        model_hp_dict, autoencoder_hp_dict
    )
    print("type(unlabeled_loader)", type(unlabeled_loader))

    # train the autoencoder first
    ae = trainer_ae(autoencoder_hp_dict, unlabeled_loader, autoencoder_loss_history)

    # then append the training on CNN from block 1
    trainer_prediction(
        model_hp_dict, ae, train_loader, valid_loader, autoencoder_loss_history
    )


def main(config_ae, config_model):

    # reads in the config files
    autoencoder_config = configparser.ConfigParser()
    autoencoder_config.read(config_ae)
    autoencoder_hp_dict = get_hyperparameters(autoencoder_config, autoencoder=True)
    print(autoencoder_hp_dict)
    model_config = configparser.ConfigParser()
    model_config.read(config_model)
    model_hp_dict = get_hyperparameters(model_config)
    print(model_hp_dict)

    # Call functions to run models
    run(autoencoder_hp_dict, model_hp_dict)


if __name__ == "__main__":

    # Read the ini file name from sys arg to avoid different people's different local set up
    # Use a shell script instead to run on your setup

    main(sys.argv[1], sys.argv[2])
