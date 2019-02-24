import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#from tensorboardX import SummaryWriter

import src.legacy.TABaseline.code.baseline_models as models
import src.legacy.TABaseline.code.scoring_function as scoreF
import src.legacy.TABaseline.code.ecgdataset as ecgdataset
from src.algorithm.autoencoder import AutoEncoder
from src.legacy.TABaseline.code import Preprocessor as pp
from src.data.unlabelled_data import UnlabelledDataset
from src.legacy.TABaseline.code.baseline_multitask_main import eval_model, train_model, training_loop, save_model, load_model
from src.algorithm.CNN_multitask import Conv1DBNLinear
from src.utils import constants

import sys
import configparser

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

# fix the seed for reproducibility
seed = 54
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

target_out_size_dict = {
    'pr_mean': 1,
    'rt_mean': 1,
    'rr_stdev': 1,
    'userid': 32,
}
target_criterion_dict = {
    'pr_mean': nn.MSELoss(),
    'rt_mean': nn.MSELoss(),
    'rr_stdev': nn.MSELoss(),
    'userid': nn.CrossEntropyLoss(),
}


targets = 'pr_mean, rt_mean, rr_stdev, userid'


def get_hyperparameters(config, autoencoder=False):
    hyperparam = {}
    hyperparam['learning_rate'] = \
        float(config.get('optimizer', 'learning_rate'))
    hyperparam['momentum'] = \
        float(config.get('optimizer', 'momentum'))
    hyperparam['batchsize'] = \
        int(config.get('optimizer', 'batch_size'))
    hyperparam['nepoch'] = \
        int(config.get('optimizer', 'nepoch'))
    hyperparam['model'] = \
        config.get('model', 'name')
    hyperparam['hidden_size'] = \
        int(config.get('model', 'hidden_size'))
    hyperparam['dropout'] = \
        float(config.get('model', 'dropout'))
    hyperparam['n_layers'] = \
        int(config.get('model', 'n_layers'))
    if not autoencoder:
        hyperparam['kernel_size'] = \
            int(config.get('model', 'kernel_size'))
        hyperparam['pool_size'] = \
            int(config.get('model', 'pool_size'))
        hyperparam['tbpath'] = \
            config.get('path', 'tensorboard')
        hyperparam['modelpath'] = \
            config.get('path', 'model')
        weight1 = float(config.get('loss', 'weight1'))
        weight2 = float(config.get('loss', 'weight2'))
        weight3 = float(config.get('loss', 'weight3'))
        weight4 = float(config.get('loss', 'weight4'))
        hyperparam['weight'] = \
            [weight1, weight2, weight3, weight4]
    return hyperparam

def train_autoencoder(epoch, model, optimizer, batch_size, train_loader):
    model.train()
    total_batch = constants.UNLABELED_SHAPE[0] // batch_size

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        data = data.view(batch_size, 1, 3750)
        # print("data iter to dev", data)

        optimizer.zero_grad()
        output = model(data)


        data = pp.Preprocessor().forward(data)
        # print("data after prepro\n")
        # print(data)
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
        # TODO: make a helper function under util/cache.py and use a name generator for the model
        # torch.save(model.state_dict(), "../../model")
    return MSE_loss.item()


if __name__ == '__main__':
    autoencoder_config = configparser.ConfigParser()
    autoencoder_config.read("src/algorithm/autoencoder_input.in")
    autoencoder_hp_dict = get_hyperparameters(autoencoder_config, autoencoder=True)
    model_config = configparser.ConfigParser()
    model_config.read("src/scripts/model_input.in")
    model_hp_dict = get_hyperparameters(model_config)

    train_dataset = ecgdataset.ECGDataset(
        constants.TRAIN_LABELED_DATASET_PATH, True, target=targets)
    valid_dataset = ecgdataset.ECGDataset(
        constants.VALID_LABELED_DATASET_PATH, False, target=targets)
    unlabeled_dataset = UnlabelledDataset(
        constants.UNLABELED_DATASET_PATH, False)

    train_loader = DataLoader(train_dataset, model_hp_dict['batchsize'], shuffle=True,
                              num_workers=1)
    valid_loader = DataLoader(valid_dataset, model_hp_dict['batchsize'], shuffle=False,
                              num_workers=1)
    unlabeled_loader = DataLoader(unlabeled_dataset, autoencoder_hp_dict['batchsize'], shuffle=False,
                              num_workers=1)

    # Autoencoder training
    autoencoder = AutoEncoder().to(device)
    AE_optimizer = optim.Adam(autoencoder.parameters(), lr=autoencoder_hp_dict['learning_rate'])

    autoencoder_loss_history = []
    for epoch in range(autoencoder_hp_dict['nepoch']):
        train_loss = train_autoencoder(
            epoch, autoencoder, AE_optimizer, autoencoder_hp_dict['batchsize'], unlabeled_loader)
        autoencoder_loss_history.append(train_loss)
    print("Autoencoder training done")

    # Model initialization
    input_size = 3750
    target_labels = targets.split(",")
    target_labels = [s.lower().strip() for s in target_labels]
    if len(target_labels) == 1:
        out_size = target_out_size_dict[target_labels[0]]
    else:
        out_size = [
            target_out_size_dict[a] for a in target_labels
        ]
    n_layers = model_hp_dict['n_layers']
    hidden_size = model_hp_dict['hidden_size']
    kernel_size = model_hp_dict['kernel_size']
    pool_size = model_hp_dict['pool_size']
    dropout = model_hp_dict['dropout']

    path = './'
    prefix = model_hp_dict['modelpath']
    chkptg_freq = 50

    # define the model
    model = Conv1DBNLinear(
        1, out_size, hidden_size, kernel_size, pool_size, dropout, autoencoder=True)
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
       lr=model_hp_dict['learning_rate'],
    )
    model.encoder.load_state_dict(autoencoder.encoder.state_dict())
    model.decoder.load_state_dict(autoencoder.decoder.state_dict())

    if len(target_labels) == 1:
        criterion = target_criterion_dict[target_labels[0]]
    else:
        criterion = [
            target_criterion_dict[a] for a in target_labels
        ]

    scoring_func_param_index = [
        None if target_labels.count(
            'pr_mean') == 0 else target_labels.index('pr_mean'),
        None if target_labels.count(
            'rt_mean') == 0 else target_labels.index('rt_mean'),
        None if target_labels.count(
            'rr_stdev') == 0 else target_labels.index('rr_stdev'),
        None if target_labels.count(
            'userid') == 0 else target_labels.index('userid'),
    ]

    # Training loop
    train, valid = training_loop(
        model, optimizer, criterion, train_loader,
        valid_loader, scoring_func_param_index,
        model_hp_dict, chkptg_freq, prefix, path
    )
    print('Done')
