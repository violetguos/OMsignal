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
from src.scripts.ecgdataset import ECGDataset
from src.legacy.TABaseline.code.baseline_models import Conv1DBNLinear
from src.legacy.TABaseline.code.baseline_multitask_main import get_hyperparameters

from src.legacy.meanteacher.pytorch.mean_teacher.data import TwoStreamBatchSampler
from src.legacy.meanteacher.pytorch.main import save_checkpoint, accuracy, update_ema_variables
from src.legacy.meanteacher.pytorch.mean_teacher.run_context import RunContext
from src.legacy.meanteacher.pytorch.mean_teacher.losses import softmax_kl_loss, softmax_mse_loss, symmetric_mse_loss
from src.legacy.meanteacher.pytorch.mean_teacher.utils import AverageMeterSet, AverageMeter
from src.legacy.meanteacher.pytorch.mean_teacher.ramps import cosine_rampdown, linear_rampup, sigmoid_rampup

from src.scripts.dataloader_utils import import_OM


# Global variables
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

def create_DataLoaders(train_data, valid_data, target='userid', batch_size = 256, labeled_batch_size = 10, include_unlabeled = True, unlabeled_data = None,
                       num_workers = 0, use_transform = True): # Need to lace training dataset with unlabeled examples
    """
    :return:
    """
    unlabeled_train_data = train_data
    if include_unlabeled:
        unlabeled_train_data = np.concatenate((train_data, unlabeled_data), axis=0)
        np.random.shuffle(unlabeled_train_data)

    unlabeled_idx = [i for i,ex in enumerate(unlabeled_train_data) if ex[:,-1]==NO_LABEL]
    labeled_idx = [i for i, ex in enumerate(unlabeled_train_data) if ex[:, -1] != NO_LABEL]

    if include_unlabeled:
        batch_sampler = TwoStreamBatchSampler(unlabeled_idx, labeled_idx, batch_size, labeled_batch_size)
    else:
        sampler = SubsetRandomSampler(labeled_idx)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)

    train_dataset = ECGDataset(unlabeled_train_data, target=target, use_transform=use_transform)
    valid_dataset = ECGDataset(valid_data, target=target, use_transform=use_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=num_workers,
                                               pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers=2 * num_workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, valid_loader

def create_ema_model(model):
    """

    :param model:
    :return:
    """
    for param in model.parameters():
        param.detach()
    return model

def get_current_consistency_weight(epoch, consistency, consistency_rampup=200):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def adjust_learning_rate(lr, initial_lr, lr_rampup, lr_rampdown_epochs, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, lr_rampup) * (lr - initial_lr) + initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if lr_rampdown_epochs:
        assert lr_rampdown_epochs >= epochs
        lr *= cosine_rampdown(epoch, lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train(train_loader, model, ema_model, optimizer, epoch, log, class_criterion = [nn.CrossEntropyLoss()], consistency_type = 'mse', weight=None, lr=1E-3,
          ema_decay=0.999, print_freq=10, consistency=True, logit_distance_cost=-0.5, initial_lr=1E-5, lr_rampup=100, lr_rampdown_epochs=550):
    """

    :param train_loader:
    :param model:
    :param ema_model:
    :param optimizer:
    :param epoch:
    :param log:
    :param class_criterion: list of losses, for each prediction task
    :param consistency_type:
    :return:
    """
    global global_step

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).to(device)
    # class_criterion_2 = class_criterion[1](size_average=False, ignore_index=NO_LABEL).cuda()
    # class_criterion_3 = class_criterion[2](size_average=False, ignore_index=NO_LABEL).cuda()
    # class_criterion_4 = class_criterion[3](size_average=False, ignore_index=NO_LABEL).cuda()
    #nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()


    if consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, consistency_type
    residual_logit_criterion = symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    ##------------ CODE FROM BASELINE --------##
    # train_loss = 0
    # train_n_iter = 0
    # if weight is None:
    #     if isinstance(criterion, (list, tuple)):
    #         weight = [1.0] * len(criterion)
    #     else:
    #         weight = 1.0
    #
    # # prMean_pred, prMean_true = None, None
    # # rtMean_pred, rtMean_true = None, None
    # # rrStd_pred, rrStd_true = None, None
    # ecgId_pred, ecgId_true = None, None

    ##------------ CODE FROM BASELINE --------##

    end = time.time()
    for i, ((input, ema_input), target) in enumerate(train_loader): ## Pète au frette
        # Measure data loading time
        meters.update('data_time', time.time() - end)

        lr = adjust_learning_rate(lr, initial_lr, lr_rampup, lr_rampdown_epochs, optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        # Variable type creation
        input_var = torch.autograd.Variable(input.to(device))
        ema_input_var = torch.autograd.Variable(ema_input.to(device))
        target_var = torch.autograd.Variable(target.to(device))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size.cpu().item())

        ###----------- CODE FROM BASELINE ---------------###
        ema_model_out = ema_model(ema_input_var)
        ema_model_out = F.softmax(ema_model_out, dim=1)
        model_out = model(input_var)
        model_out = F.softmax(model_out, dim=1)


        ##------------- CODE FROM BASELINE -------------###
        if isinstance(model_out, Variable):
            assert logit_distance_cost < 0
            logit1 = model_out
            ema_logit = ema_model_out
        else:
            assert len(model_out) == 2
            assert len(ema_model_out) == 2
            logit1, logit2 = model_out
            ema_logit, _ = ema_model_out

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        if logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.cpu().item())
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0

        class_loss = class_criterion(class_logit, target_var.squeeze()) / minibatch_size
        meters.update('class_loss', class_loss.cpu().item())

        ema_class_loss = class_criterion(ema_logit, target_var.squeeze()) / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.cpu().item())

        if consistency:
            consistency_weight = get_current_consistency_weight(epoch, consistency)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.cpu().item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        loss = class_loss + consistency_loss + res_loss
        assert not (np.isnan(loss.data.cpu()) or loss.data.cpu() > 1e5), 'Loss explosion: {}'.format(loss.data)
        meters.update('loss', loss.cpu().item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('error1', 100. - prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('top5', prec5[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('error5', 100. - prec5[0].cpu().item(), labeled_minibatch_size.cpu().item())

        ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
        meters.update('ema_top1', ema_prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('ema_error1', 100. - ema_prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('ema_top5', ema_prec5[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('ema_error5', 100. - ema_prec5[0].cpu().item(), labeled_minibatch_size.cpu().item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })

def validate(eval_loader, model, log, global_step, epoch, print_freq = 10):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, ((input, _), target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input.to(device))
        target_var = torch.autograd.Variable(target.to(device))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size.cpu().item())

        # compute output
        output1 = model(input_var)
        softmax1 = F.softmax(output1, dim=1)
        class_loss = class_criterion(softmax1, target_var.squeeze()) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('top1', prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('error1', 100.0 - prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('top5', prec5[0].cpu().item(), labeled_minibatch_size.cpu().item())
        meters.update('error5', 100.0 - prec5[0].cpu().item(), labeled_minibatch_size.cpu().item())

        # measure elapsed time
        meters.update('batch_time', time.time() - end)

        test1 = class_loss.cpu().item()
        test2 = prec1[0].cpu().item()

        end = time.time()

        if i % print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    i, len(eval_loader), meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    return meters['top1'].avg

#########################################################################################
#########################################################################################

if __name__ == "__main__":

    # Setting up log
    logging.basicConfig(level=logging.INFO)

    context = RunContext(__file__,0)

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")


    # Setting up GPU
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device enabled: {}".format(device))

    # Get hyperparameters
    config = configparser.ConfigParser()
    config.sections()
    config.read('src/scripts/mean_teacher/configuration.ini')
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
        model_ema = models.LSTMLinear(
            input_size, out_size, hidden_size, n_layers, dropout)
        model_ema = create_ema_model(model_ema)
    elif modeltype == 'RNN':
        model = models.RNNLinear(input_size, out_size,
                                 hidden_size, n_layers, dropout)
        model_ema = models.RNNLinear(input_size, out_size,
                                 hidden_size, n_layers, dropout)
        model_ema = create_ema_model(model_ema)
    elif modeltype == 'MLP':
        model = models.MLP(input_size, out_size, hidden_size)
        model_ema = models.MLP(input_size, out_size, hidden_size)
        model_ema = create_ema_model(model_ema)
    elif modeltype == 'CONV1D':
        model = models.Conv1DLinear(
            1, out_size, hidden_size, kernel_size, pool_size)
        model_ema = models.Conv1DLinear(
            1, out_size, hidden_size, kernel_size, pool_size)
        model_ema = create_ema_model(model_ema)
    elif modeltype == 'CONV1DBN':
        model = models.Conv1DBNLinear(
            1, out_size, hidden_size, kernel_size, pool_size, dropout)
        model_ema = models.Conv1DBNLinear(
            1, out_size, hidden_size, kernel_size, pool_size, dropout)
        model_ema = create_ema_model(model_ema)
    else:
        print('Model should be set to LSTM/RNN/MLP/CONV1D/CONV1DBN')
        exit()

    # Model to gpu, Create optimizer, criterion and train
    model.to(device)
    model_ema.to(device)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=w_d)

    if len(target_labels) == 1:
        criterion = target_criterion_dict[target_labels[0]]
    else:
        criterion = [
            target_criterion_dict[a] for a in target_labels
        ]

    scoring_func_param_index = [
        # None if target_labels.count(
        #     'pr_mean') == 0 else target_labels.index('pr_mean'),
        # None if target_labels.count(
        #     'rt_mean') == 0 else target_labels.index('rt_mean'),
        # None if target_labels.count(
        #     'rr_stdev') == 0 else target_labels.index('rr_stdev'),
        None if target_labels.count(
            'userid') == 0 else target_labels.index('userid'),
    ]
    evaluation_epochs = 1
    checkpoint_epochs = 5

    for epoch in range(epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, model_ema, optimizer, epoch, training_log, consistency_type='mse',
              lr=lr, initial_lr=0, lr_rampup=100, ema_decay=0.999)
        LOG.info("--- training epoch %s in %s seconds ---" % (epoch, time.time() - start_time))

        if evaluation_epochs and (epoch + 1) % evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(valid_loader, model, validation_log, global_step, epoch + 1)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(valid_loader, model_ema, ema_validation_log, global_step, epoch + 1)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if checkpoint_epochs and (epoch + 1) % checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': 'CNNDBN',
                'state_dict': model.state_dict(),
                'ema_state_dict': model_ema.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)