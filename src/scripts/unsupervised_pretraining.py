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


def eval_model(autoencoder, prediction_module, criterion, eval_loader, score_param_index, weight=None):
    """
    :param autoencoder:
    :param prediction_module:
    :param criterion:
    :param eval_loader:
    :param score_param_index:
    :param weight:
    :return: mean_loss, metrics
    """
    # TODO Write the function


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

def train_prediction_module(autoencoder, model, optimizer, criterion, train_loader, score_param_index,
                weight=None):
    model.train()
    train_loss = 0
    train_n_iter = 0
    correct_predictions = 0
    if weight is None:
        if isinstance(criterion, (list, tuple)):
            weight = [1.0] * len(criterion)
        else:
            weight = 1.0

    prMean_pred, prMean_true = None, None
    rtMean_pred, rtMean_true = None, None
    rrStd_pred, rrStd_true = None, None
    ecgId_pred, ecgId_true = None, None

    for x, y in train_loader:
        x = x.to(device)
        x = autoencoder.encoder(x)
        if isinstance(y, (list, tuple)):
            y = [t.to(device) for t in y]
        else:
            y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)

        if isinstance(outputs, (list, tuple)):
            loss = [
                w * c(o, gt) if o.size(1) == 1
                else w * c(o, gt.squeeze(1))
                for c, o, gt, w in zip(criterion, outputs, y, weight)
            ]
            loss = sum(loss)

            if score_param_index[0] is not None:
                i = score_param_index[0]
                if prMean_true is None:
                    prMean_true = y[i].view(-1).tolist()
                    prMean_pred = outputs[i].view(-1).tolist()
                else:
                    prMean_true.extend(y[i].view(-1).tolist())
                    prMean_pred.extend(outputs[i].view(-1).tolist())
            if score_param_index[1] is not None:
                i = score_param_index[1]
                if rtMean_true is None:
                    rtMean_true = y[i].view(-1).tolist()
                    rtMean_pred = outputs[i].view(-1).tolist()
                else:
                    rtMean_true.extend(y[i].view(-1).tolist())
                    rtMean_pred.extend(outputs[i].view(-1).tolist())
            if score_param_index[2] is not None:
                i = score_param_index[2]
                if rrStd_true is None:
                    rrStd_true = y[i].view(-1).tolist()
                    rrStd_pred = outputs[i].view(-1).tolist()
                else:
                    rrStd_true.extend(y[i].view(-1).tolist())
                    rrStd_pred.extend(outputs[i].view(-1).tolist())
            if score_param_index[3] is not None:
                i = score_param_index[3]
                _, pred_classes = torch.max(outputs[i], dim=1)
                if ecgId_true is None:
                    ecgId_true = y[i].view(-1).tolist()
                    ecgId_pred = pred_classes.view(-1).tolist()
                else:
                    ecgId_true.extend(y[i].view(-1).tolist())
                    ecgId_pred.extend(pred_classes.view(-1).tolist())

        else:
            loss = criterion(outputs, y.squeeze(1))

            if score_param_index[0] is not None:
                if prMean_true is None:
                    prMean_true = y.view(-1).tolist()
                    prMean_pred = outputs.view(-1).tolist()
                else:
                    prMean_true.extend(y.view(-1).tolist())
                    prMean_pred.extend(outputs.view(-1).tolist())
            elif score_param_index[1] is not None:
                if rtMean_true is None:
                    rtMean_true = y.view(-1).tolist()
                    rtMean_pred = outputs.view(-1).tolist()
                else:
                    rtMean_true.extend(y.view(-1).tolist())
                    rtMean_pred.extend(outputs.view(-1).tolist())
            elif score_param_index[2] is not None:
                if rrStd_true is None:
                    rrStd_true = y.view(-1).tolist()
                    rrStd_pred = outputs.view(-1).tolist()
                else:
                    rrStd_true.extend(y.view(-1).tolist())
                    rrStd_pred.extend(outputs.view(-1).tolist())
            elif score_param_index[3] is not None:
                _, pred_classes = torch.max(outputs, dim=1)
                if ecgId_true is None:
                    ecgId_true = y.view(-1).tolist()
                    ecgId_pred = pred_classes.view(-1).tolist()
                else:
                    ecgId_true.extend(y.view(-1).tolist())
                    ecgId_pred.extend(pred_classes.view(-1).tolist())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_n_iter += 1

    # metrics
    prMean_pred = None if prMean_pred is None else np.array(
        prMean_pred, dtype=np.float32)
    prMean_true = None if prMean_true is None else np.array(
        prMean_true, dtype=np.float32)
    rtMean_pred = None if rtMean_pred is None else np.array(
        rtMean_pred, dtype=np.float32)
    rtMean_true = None if rtMean_true is None else np.array(
        rtMean_true, dtype=np.float32)
    rrStd_pred = None if rrStd_pred is None else np.array(
        rrStd_pred, dtype=np.float32)
    rrStd_true = None if rrStd_true is None else np.array(
        rrStd_true, dtype=np.float32)
    ecgId_pred = None if ecgId_pred is None else np.array(
        ecgId_pred, dtype=np.int32)
    ecgId_true = None if ecgId_true is None else np.array(
        ecgId_true, dtype=np.int32)

    metrics = scoreF.scorePerformance(
        prMean_pred, prMean_true,
        rtMean_pred, rtMean_true,
        rrStd_pred, rrStd_true,
        ecgId_pred, ecgId_true
    )

    # mean loss
    mean_loss = train_loss / max(train_n_iter, 1)

    return mean_loss, metrics


def save_model(epoch, model, prefix='neural_network', path='./'):

    # creating a filename indexed by the epoch value
    filename = path + prefix + '_{}.pt'.format(epoch)

    # save the parameters of the model.
    torch.save(model.state_dict(), filename)


def load_model(epoch, model, prefix='neural_network', path='./'):

    # creating a filename indexed by the epoch value
    filename = path + prefix + '_{}.pt'.format(epoch)

    # load parameters from saved model
    model.load_state_dict(torch.load(filename))

    return model


def training_loop(
        autoencoder,
        model,
        AE_optimizer,
        model_optimizer,
        criterion,
        train_loader,
        unlabeled_loader,
        eval_loader,
        score_param_index,
        autoencoder_hp_dict,
        model_hp_dict,
        chkptg_freq=10,
        prefix='neural_network',
        path='./'):
    # train the model using optimizer / criterion
    # this function also creates a tensorboard log
    # writer = SummaryWriter(hyperparameters_dict['tbpath'])

    autoencoder_loss_history = []
    for epoch in range(autoencoder_hp_dict['nepoch']):
        train_loss = train_autoencoder(
            epoch, autoencoder, AE_optimizer, autoencoder_hp_dict['batchsize'], unlabeled_loader)
        autoencoder_loss_history.append(train_loss)
    print("Autoencoder training done")

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    weight = model_hp_dict['weight']

    for epoch in range(1, model_hp_dict['nepoch'] + 1):
        train_loss, train_acc = train_prediction_module(
            autoencoder, model, model_optimizer, criterion,
            train_loader, score_param_index, weight
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

       # #valid_loss, valid_acc = eval_model(
       #     model, criterion, eval_loader,
       #     score_param_index, weight
       # )
       #valid_loss_history.append(valid_loss)
       #valid_acc_history.append(valid_acc)

       #writer.add_scalar('Training/Loss', train_loss, epoch)
       #writer.add_scalar('Valid/Loss', valid_loss, epoch)

       #writer.add_scalar('Training/OverallScore', train_acc[0], epoch)
       #writer.add_scalar('Valid/OverallScore', valid_acc[0], epoch)

       #writer.add_scalar('Training/prMeanTau', train_acc[1], epoch)
       #writer.add_scalar('Valid/prMeanTau', valid_acc[1], epoch)

       #writer.add_scalar('Training/rtMeanTau', train_acc[2], epoch)
       #writer.add_scalar('Valid/rtMeanTau', valid_acc[2], epoch)

       #writer.add_scalar('Training/rrStdDevTau', train_acc[3], epoch)
       #writer.add_scalar('Valid/rrStdDevTau', valid_acc[3], epoch)

       #writer.add_scalar('Training/userIdAcc', train_acc[4], epoch)
       #writer.add_scalar('Valid/userIdAcc', valid_acc[4], epoch)

        print("Epoch {} {} {}".format(
            #epoch, train_loss, valid_loss, train_acc, valid_acc)
            epoch, train_loss, train_acc)
        )

        # Checkpoint
    #    if epoch % chkptg_freq == 0:
    #        save_model(epoch, model, prefix, path)
    #save_model(model_hp_dict['nepoch'], model, prefix, path)
    return [
        (train_loss_history, train_acc_history),
        (0,0)
        # (valid_loss_history, valid_acc_history)
    ]


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

    # Autoencoder initialization
    autoencoder = AutoEncoder().to(device)
    AE_optimizer = optim.Adam(autoencoder.parameters(), lr=autoencoder_hp_dict['learning_rate'])

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
    model = models.Conv1DBNLinear(
        1, out_size, hidden_size, kernel_size, pool_size, dropout)
    # model to gpu, create optimizer, criterion and train
    model.to(device)
    model_optimizer = optim.Adam(model.parameters(),
                           lr=model_hp_dict['learning_rate'])

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
        autoencoder, model, AE_optimizer, model_optimizer, criterion,
        train_loader, unlabeled_loader, valid_loader,
        scoring_func_param_index, autoencoder_hp_dict, model_hp_dict,
        chkptg_freq, prefix, path
    )
    print('Done')
