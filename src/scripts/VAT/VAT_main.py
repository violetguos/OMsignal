import os
import sys
import time
import logging
import configparser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import src.legacy.TABaseline.code.baseline_models as models
import src.legacy.TABaseline.code.scoring_function as scoreF
from src.legacy.TABaseline.code.baseline_models import Conv1DBNLinear

from src.legacy.meanteacher.pytorch.mean_teacher.data import TwoStreamBatchSampler
from src.legacy.meanteacher.pytorch.main import save_checkpoint, accuracy, update_ema_variables
from src.legacy.meanteacher.pytorch.mean_teacher.run_context import RunContext
from src.legacy.meanteacher.pytorch.mean_teacher.losses import softmax_kl_loss, softmax_mse_loss, symmetric_mse_loss
from src.legacy.meanteacher.pytorch.mean_teacher.utils import AverageMeterSet, AverageMeter
from src.legacy.meanteacher.pytorch.mean_teacher.ramps import cosine_rampdown, linear_rampup, sigmoid_rampup

from src.legacy.VAT.utils import vat_loss, entropy_loss

from src.legacy.VAT_2.vat import VATLoss

from src.scripts.dataloader_utils import import_OM
from src.scripts.mean_teacher.mean_teacher_main import create_DataLoaders
from src.scripts.ecgdataset import ECGDataset

# Global Variables ############

targets = 'userid' # 'pr_mean, rt_mean, rr_stdev'
NO_LABEL = -1
args = None
best_prec1 = 0
global_step = 0
LOG = logging.getLogger('main')

target_criterion_dict = {
    # 'pr_mean': nn.MSELoss(),
    # 'rt_mean': nn.MSELoss(),
    # 'rr_stdev': nn.MSELoss(),
    'userid': nn.CrossEntropyLoss()
}

target_out_size_dict = {
    # 'pr_mean': 1,
    # 'rt_mean': 1,
    # 'rr_stdev': 1,
    'userid': 32,
}


def get_hyperparameters(config):
    # from a ConfigParser() file; get hyperparameters
    # return them in a dict
    hyperparam = {}
    hyperparam['learning_rate'] = \
        float(config.get('optimizer', 'learning_rate'))
    hyperparam['momentum'] = \
        float(config.get('optimizer', 'momentum'))
    hyperparam['batchsize'] = \
        int(config.get('optimizer', 'batch_size'))
    hyperparam['nepoch'] = \
        int(config.get('optimizer', 'nepoch'))
    hyperparam['weight_decay'] = \
        float(config.get('optimizer', 'weight_decay'))
    # hyperparam['milestones'] = \
    #     (config.get('optimizer', 'milestones'))
    hyperparam['milestone_shrink'] = \
        float(config.get('optimizer', 'milestone_shrink'))
    hyperparam['model'] = \
        config.get('model', 'name')
    hyperparam['hidden_size'] = \
        int(config.get('model', 'hidden_size'))
    hyperparam['dropout'] = \
        float(config.get('model', 'dropout'))
    hyperparam['n_layers'] = \
        int(config.get('model', 'n_layers'))
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

def train_VAT(model, sample, target, optimizer, loss = nn.CrossEntropyLoss(), entropy=True, alpha = 1):

    model.train()
    labeled_sample = torch.tensor(np.array([sample[i].cpu().numpy() for i,ex in enumerate(sample) if target[i].item() != 0], dtype=np.float32), requires_grad = True).to(device)
    unlabeled_sample = torch.tensor(np.array([sample[i].cpu().numpy() for i,ex in enumerate(sample) if target[i].item() == 0], dtype=np.float32), requires_grad = True).to(device)
    true_targets = torch.tensor(np.array([target[i].cpu().item() for i,ex in enumerate(sample) if target[i].item() != 0], dtype=np.float32), requires_grad = True).to(device)

    optimizer.zero_grad()
    VATLoss_ = VATLoss()
    vat_loss_ = VATLoss_(model, unlabeled_sample) / len(unlabeled_sample)

    output = model(labeled_sample)
    output_unlabeled = model(unlabeled_sample)

    acc = np.array(accuracy(F.softmax(output, dim=1), true_targets))
    meters.update('accuracy', acc)

    loss_ = loss(output, true_targets.long()) / len(labeled_sample)

    total_loss = loss_ + alpha * vat_loss_
    if entropy:
        entropy_loss_ = entropy_loss(output_unlabeled)
        total_loss += entropy_loss_

    meters.update('loss', loss_.cpu().item(), np.shape(labeled_sample)[0])
    meters.update('vat_loss', vat_loss_.cpu().item(), np.shape(unlabeled_sample)[0])
    meters.update('entropy_loss', entropy_loss_.cpu().item(), np.shape(unlabeled_sample)[0])

    LOG.info(
        'Loss {meters[loss]:.4f}\t'
        'VAT Loss {meters[vat_loss]:.4f}\t'
        'Entropy Loss {meters[entropy_loss]:.3f}\t'
        'Accuracy {meters[accuracy]:.3f}'.format(meters=meters))

    return total_loss, loss_, vat_loss_, entropy_loss

def validate_VAT(model, sample, target, loss = nn.CrossEntropyLoss()):

    model.eval()
    output = F.softmax(model(sample), dim=1)
    acc = accuracy(torch.max(output, dim=1)[1], target)

    return acc

###################################


if __name__ == "__main__":

    # Setting up logging objects
    logging.basicConfig(level=logging.INFO)
    context = RunContext(__file__,0)

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    meters = AverageMeterSet()


    # Setting up GPU
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device enabled: {}".format(device))

    # Get hyperparameters
    config = configparser.ConfigParser()
    config.read('src/scripts/VAT/configuration.ini')
    hyperparameters_dict = get_hyperparameters(config)

    n_layers = hyperparameters_dict['n_layers']
    hidden_size = hyperparameters_dict['hidden_size']
    kernel_size = hyperparameters_dict['kernel_size']
    pool_size = hyperparameters_dict['pool_size']
    dropout = hyperparameters_dict['dropout']
    batchsize = hyperparameters_dict['batchsize']
    epochs = hyperparameters_dict['nepoch']
    lr = hyperparameters_dict['learning_rate']
    w_d = hyperparameters_dict['weight_decay']
    # milestones = hyperparameters_dict['milestones']
    milestones = [0.2, 0.4, 0.6, 0.8]
    milestone_shrink = hyperparameters_dict['milestone_shrink']

    input_size = 3750
    target_labels = targets.split(",")
    target_labels = [s.lower().strip() for s in target_labels]
    if len(target_labels) == 1:
        out_size = target_out_size_dict[target_labels[0]]
    else:
        out_size = [
            target_out_size_dict[a] for a in target_labels
        ]

    # Fetch data -- {}_dataset is a (N_samples, 2) shape vector
    train_data, tr_pr, tr_rt, tr_rr, tr_ids = import_OM("training", cluster=False)
    train_data = np.concatenate((np.array(train_data),
                                 np.array(tr_pr)[:,np.newaxis],
                                 np.array(tr_rt)[:,np.newaxis],
                                 np.array(tr_rr)[:,np.newaxis],
                                 np.array(tr_ids)[:,np.newaxis]), axis=2)

    valid_data, va_pr, va_rt, va_rr, va_ids = import_OM("validation", cluster=False)
    valid_data = np.concatenate((np.array(valid_data),
                                 np.array(va_pr)[:,np.newaxis],
                                 np.array(va_rt)[:,np.newaxis],
                                 np.array(va_rr)[:,np.newaxis],
                                 np.array(va_ids)[:,np.newaxis]), axis=2)

    unlabeled_data = import_OM("unlabeled", cluster=False, len=10000)
    unlabeled_data = np.concatenate((np.array(unlabeled_data)[:,np.newaxis,:],
                                     np.float(NO_LABEL) *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis],
                                     np.float(NO_LABEL) *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis],
                                     np.float(NO_LABEL) *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis],
                                     NO_LABEL *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis]), axis=2)

    # Create DataLoaders
    train_loader, valid_loader = create_DataLoaders(train_data, valid_data, include_unlabeled=True,
                                                    unlabeled_data=unlabeled_data, num_workers=0, batch_size=batchsize)

    # Define and Instanciate the model
    modeltype = hyperparameters_dict['model']
    if modeltype == 'LSTM':
        model = models.LSTMLinear(
            input_size, out_size, hidden_size, n_layers, dropout)
    elif modeltype == 'RNN':
        model = models.RNNLinear(input_size, out_size,
                                 hidden_size, n_layers, dropout)
    elif modeltype == 'MLP':
        model = models.MLP(input_size, out_size, hidden_size)
    elif modeltype == 'CONV1D':
        model = models.Conv1DLinear(
            1, out_size, hidden_size, kernel_size, pool_size)
    elif modeltype == 'CONV1DBN':
        model = models.Conv1DBNLinear(
            1, out_size, hidden_size, kernel_size, pool_size, dropout)
    else:
        print('Model should be set to LSTM/RNN/MLP/CONV1D/CONV1DBN')
        exit()

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=w_d)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, (np.array(milestones).dot(epochs)), gamma=milestone_shrink)

    for epoch in range(epochs):

        scheduler.step(epoch)
        for idx, ((sample , _), target) in enumerate(train_loader):
            sample, target = sample.to(device), target.to(device)
            LOG.info('Epoch: [{0}][{1}/{2}]\t -------------------'.format(epoch, idx, len(train_loader)))
            total_loss, CE_loss, vat_loss, entropy_loss = train_VAT(model, sample, target, optimizer)
            total_loss.backward()
            optimizer.step()

        LOG.info('Validation ----------------------')
        for idx, (sample, target) in enumerate(valid_loader):
            acc = []
            acc_ = validate_VAT(model, sample, target)
            acc.append(acc_)
        LOG.info('Accuracy: {acc}'.format(acc = np.mean(acc)))



    # for epoch in range(opt.num_epochs):
    #
    #     if epoch > opt.epoch_decay_start:
    #         decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
    #         optimizer.lr = decayed_lr
    #         optimizer.betas = (0.5, 0.999)
    #
    #     for i in range(num_iter_per_epoch):
    #
    #         batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
    #         x = labeled_train[batch_indices]
    #         y = labeled_target[batch_indices]
    #         batch_indices_unlabeled = torch.LongTensor(np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
    #         ul_x = unlabeled_train[batch_indices_unlabeled]
    #
    #         v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
    #                                 optimizer)
    #
    #         if i % 100 == 0:
    #             print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.data[0], "CE Loss :", ce_loss.data[0])
    #
    #     if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:
    #
    #         batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
    #         x = labeled_train[batch_indices]
    #         y = labeled_target[batch_indices]
    #         train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
    #         print("Train accuracy :", train_accuracy.data[0])
    #
    #         for (data, target) in test_loader:
    #             test_accuracy = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    #             print("Test accuracy :", test_accuracy.data[0])
    #             break
    #
    #
    # test_accuracy = 0.0
    # counter = 0
    # for (data, target) in test_loader:
    #     n = data.size()[0]
    #     acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    #     test_accuracy += n*acc
    #     counter += n
    #
    # print("Full test accuracy :", test_accuracy.data[0]/counter)