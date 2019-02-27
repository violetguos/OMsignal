import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.legacy.TABaseline.code.baseline_models import Conv1DBNLinear
import src.legacy.TABaseline.code.ecgdataset as ecgdataset
from src.legacy.TABaseline.code.baseline_multitask_main import training_loop
from src.utils import constants
from src.utils.os_helper import get_hyperparameters
from src.utils.cache import ModelCache
import sys
import configparser
import os


"""
Runs the Block 1 Baseline CNN for a small number of epochs for debugging purposes
Not actually used for Block 2 semisupervised learning
"""

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


def trainer_prediction(model_hp_dict, train_loader, valid_loader, loss_history):

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
    model_hp_dict["tbpath"] = os.path.join(path, "tensorboard")
    chkptg_freq = model_hp_dict["nepoch"] // 10

    # define the model
    model = Conv1DBNLinear(1, out_size, hidden_size, kernel_size, pool_size, dropout)
    # model to gpu, create optimizer, criterion and train
    model.to(device)
    optimizer = optim.Adam(
        [
            # {"params": model.encoder.parameters(), "lr": 0},
            # {"params": model.decoder.parameters(), "lr": 0},
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

    train_loader = DataLoader(
        train_dataset, model_hp_dict["batchsize"], shuffle=True, num_workers=1
    )
    valid_loader = DataLoader(
        valid_dataset, model_hp_dict["batchsize"], shuffle=False, num_workers=1
    )
    return train_loader, valid_loader


def run(model_hp_dict):
    loss_history = ModelCache(_prefix="autoencoder", _mode="train")

    train_loader, valid_loader = load_data(model_hp_dict)

    # train the autoencoder first

    # then append the training on CNN from block 1
    trainer_prediction(model_hp_dict, train_loader, valid_loader, loss_history)


def main(config_model):

    model_config = configparser.ConfigParser()
    model_config.read(config_model)
    model_hp_dict = get_hyperparameters(model_config)

    # Call functions to run models
    run(model_hp_dict)


if __name__ == "__main__":

    # Read the ini file name from sys arg to avoid different people's different local set up
    # Use a shell script instead to run on your setup

    main(sys.argv[1])
