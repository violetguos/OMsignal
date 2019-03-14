import os
import time
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from tensorboardX import SummaryWriter
from scipy.stats import kendalltau

from src.utils.mean_teacher_data_utils import create_DataLoaders, AverageMeterSet, RunContext

## Global variables :: some of them will be found in main codes ##
LOG = logging.getLogger('main')
NO_LABEL = 0
global_step = 0
best_precision = 0
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

target_criterion_dict = {
    'pr_mean': nn.MSELoss(),
    'rt_mean': nn.MSELoss(),
    'rr_stdev': nn.MSELoss(),
    'userid': nn.CrossEntropyLoss(size_average = True, reduction='sum', ignore_index=NO_LABEL)
}
## Functions ##

def task_training(task, model, model_ema, train_data, valid_data, unlabeled_data, augment=True, batchsize=32, labeled_batch_size = 16,
                  consistency_type='mse', consistency=True, consistency_rampup=50, learning_rate=0.00, initial_lr=1E-6, lr_rampup=10,
                  lr_rampdown = 100, weight_decay=1E-4, EMA_decay=0.999, epochs=100, device=device,
                  evaluation_epochs=1, checkpoint_epochs = 5, tbpath = "Tensorboard/mean_teacher"):
    """
    Task training function. Takes as arguments the task for which to train the network. Almost all the following arguments
    are hyperparameters
    :param task: (string) task to train for: pr_mean, rt_mean, rr_stdev, userid
    :param model: (nn.Module) student model to train
    :param model_ema: (nn.Module) teacher model to train
    :param train_data: (np.array) train data
    :param valid_data: (np.array) validation data
    :param unlabeled_data: (np.array) unlabeled_data
    :param augment: (bool) flag to allow data augmentation or not
    :param batchsize: (int)
    :param labeled_batch_size: (int) amount of labeled samples within a training batch
    :param consistency_type: (string) type of loss to apply to consistency, 'mse' or 'kl'
    :param consistency: (bool) apply consistency loss or not
    :param consistency_rampup: (int) number of epochs taken for consistency weight to reach full value (1). Follow a sigmoid rampup
    :param learning_rate: (float)
    :param initial_lr: (float) initial learning rate. The l.r. will increase linearly to the value (learning_rate)
    :param lr_rampup: (int) epochs taken for the initial_lr to reach learning_rate
    :param lr_rampdown: (int) factor of linear rate decrease with epochs
    :param weight_decay: (float) weight decay
    :param EMA_decay: (float) exponentially moving average decay
    :param epochs:(int) total number of epochs
    :param device: (string) device to use
    :param evaluation_epochs: (intervals at which evaluate the training)
    :param checkpoint_epochs: (intervals at which save the model)
    :param tbpath: Tensorboard folder path
    :return: None
    """
    writer = SummaryWriter(tbpath)
    context = RunContext(__file__,0)
    global global_step
    global_step = 0
    best_precision = 0

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training_{}".format(task))
    validation_log = context.create_train_log("validation_{}".format(task))
    ema_validation_log = context.create_train_log("ema_validation_{}".format(task))

    class_criterion = target_criterion_dict[task]

    train_loader, valid_loader = create_DataLoaders(train_data, valid_data, include_unlabeled=True,
                                                    unlabeled_data=unlabeled_data,
                                                    batch_size=batchsize, labeled_batch_size=labeled_batch_size,
                                                    task=task, use_transform=augment)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    model_ema.to(device)
    for epoch in range(epochs):

        start_time = time.time()
        # train for one epoch
        train(train_loader, model, model_ema, optimizer, epoch, epochs, training_log,
              task=task, lr=learning_rate, initial_lr=initial_lr, lr_rampup=lr_rampup, lr_rampdown_epochs=lr_rampdown,
              ema_decay=EMA_decay, device=device, global_step=global_step, tb_writer=writer,
              consistency_type=consistency_type, consistency=consistency, consistency_rampup=consistency_rampup)

        LOG.info("--- training epoch %s in %s seconds ---" % (epoch, time.time() - start_time))

        if evaluation_epochs and (epoch + 1) % evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(valid_loader, model, 'Student', validation_log, global_step, epoch + 1,
                             device=device, task=task, writer=writer)
            writer.add_scalar("Student_model_validation", prec1, epoch)

            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(valid_loader, model_ema, 'Teacher', ema_validation_log, global_step, epoch + 1,
                                 device=device, task=task, writer=writer)
            writer.add_scalar("Teacher_model_validation", ema_prec1, epoch)

            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_precision
            best_precision = max(ema_prec1, best_precision)
            writer.add_scalar("Best_teacher_precision_so_far", best_precision, epoch)
        else:
            is_best = False

        if checkpoint_epochs and (epoch + 1) % checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': '{}'.format(type(model)),
                'state_dict': model.state_dict(),
                'ema_state_dict': model_ema.state_dict(),
                'best_precision': best_precision,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)

    global_step = 0
    return None

def train(train_loader, model, ema_model, optimizer, epoch, epochs, log,
          task = 'userid', consistency_type = 'mse', lr=1E-3, ema_decay=0.999, print_freq=10, consistency=True,
          consistency_rampup=50, initial_lr=1E-5, lr_rampup=100, lr_rampdown_epochs=1500, device='cuda:0',
          global_step=global_step, tb_writer = SummaryWriter()):
    """

    :param train_loader: (DataLoader) train dataloader
    :param model: (nn.Module) student model
    :param ema_model: (nn.Module) teacher module
    :param optimizer: (optimizer) optimizer to use
    :param epoch: (int) current training epoch
    :param epochs: (int) total training epochs
    :param log: (TrainLog) log where to print the train performance data
    :param task: (string) task trained on
    :param consistency_type: (string) 'mse' or 'kl', consistency loss func
    :param lr: (float) loss function
    :param ema_decay: (float) exponentially moving average decay
    :param print_freq: (int) frequency with which to print training information
    :param consistency: (bool) consistency applied or not
    :param consistency_rampup: (int) length during which increase consistency_weight to its maximum value
    :param initial_lr: (float) initial learning rate
    :param lr_rampup: (int) epochs taken for the initial_lr to reach learning_rate
    :param lr_rampdown_epochs: (int) factor of linear rate decrease with epochs
    :param device: (string) device to use
    :param global_step: (int) each step represents a call to loss.backward() and optimizer.step()
    :param tb_writer: SummaryWriter object, linked to TensorBoard
    :return: None
    """

    # Assertions
    assert task in target_criterion_dict.keys(), "Incorrect task!"
    class_criterion = target_criterion_dict[task]

    if consistency_type == 'mse':
        if task != 'userid':
            consistency_criterion = F.mse_loss
        else:
            consistency_criterion = softmax_mse_loss
    elif consistency_type == 'kl':
        if task != 'userid':
            consistency_criterion = F.kl_div
        else:
            consistency_criterion = softmax_kl_loss
    else:
        assert False, "Not a valid consistency type"

    # MeterSet for information logging
    meters = AverageMeterSet()

    # Switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i, ((input, ema_input), target) in enumerate(train_loader):

        # Measure data loading time
        input_check, ema_check = input[0].numpy(), ema_input[0].numpy()
        diff_check = input_check - ema_check
        meters.update('data_time', time.time() - end)

        lr = adjust_learning_rate(lr, initial_lr, lr_rampup, lr_rampdown_epochs, optimizer, epoch, epochs,
                                  i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])


        # Variable type creation
        input_var = torch.autograd.Variable(input.to(device))
        ema_input_var = torch.autograd.Variable(ema_input.to(device))
        target_var = torch.autograd.Variable(target.to(device))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size.cpu().item())

        ema_model_out = ema_model(ema_input_var)
        ema_softmax_out = F.softmax(ema_model_out, dim=1)

        model_out = model(input_var)
        softmax_out = F.softmax(model_out, dim=1)

        logit1 = softmax_out
        ema_logit = ema_softmax_out
        ema_cons_logit = ema_model_out

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        class_logit, cons_logit = logit1, model_out

        if task != 'userid':
            labeled_input = torch.tensor(
                np.array([input[i].cpu().numpy() for i,ex in enumerate(input) if target[i].item() != -1.0], dtype=np.float32),
                         requires_grad = True).to(device)
            ema_labeled_input = torch.tensor(
                np.array([ema_input[i].cpu().numpy() for i,ex in enumerate(input) if target[i].item() != -1.0], dtype=np.float32),
                         requires_grad = True).to(device)
            target_w_label = torch.tensor(
                np.array([target[i].cpu().numpy() for i, ex in enumerate(input) if target[i].item() != -1.0],
                         dtype=np.float32), requires_grad=True).to(device)

            model_out =  model(labeled_input) # No softmax
            class_loss = class_criterion(model_out, target_w_label) # MSE error already performs mean
            perfo_score, _ = kendalltau(model_out.detach().cpu().numpy(), target_w_label.detach().cpu().numpy())
            meters.update('class_loss', class_loss.cpu().item())
            meters.update('kentau', perfo_score)

            ema_model_out = model(ema_labeled_input)
            ema_logit_mse = ema_model_out.detach().data
            ema_logit_mse.requires_grad = False
            ema_class_loss = class_criterion(ema_logit_mse, target_w_label) # MSE error already performs mean
            ema_perfo_score, _ = kendalltau(ema_model_out.detach().cpu().numpy(), target_w_label.detach().cpu().numpy())
            meters.update('ema_kentau', ema_perfo_score)

        else: #CrossEntropyLoss
            class_loss = class_criterion(class_logit, target_var.squeeze())
            meters.update('class_loss', class_loss.cpu().item())

            ema_class_loss = class_criterion(ema_logit, target_var.squeeze())
            meters.update('ema_class_loss', ema_class_loss.cpu().item())

        if consistency:
            consistency_weight = get_current_consistency_weight(epoch, consistency, consistency_rampup)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_cons_logit)
            meters.update('cons_loss', consistency_loss.cpu().item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        loss = class_loss + consistency_loss
        if (np.isnan(loss.data.cpu()) or loss.data.cpu() > 1e5):
            breakpoint = True
            x_ = loss
            x_data = loss.data
            x_data_cpu = loss.data.cpu()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print (name, param.data)
            for name, param in ema_model.named_parameters():
                if param.requires_grad:
                    print (name, param.data)
        assert not (np.isnan(loss.data.cpu()) or loss.data.cpu() > 1e5), 'Loss explosion: {}'.format(loss.data)
        meters.update('loss', loss.cpu().item())

        if task != 'userid':

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
                    'KendallTau Student {meters[kentau]:.4f}\t'
                    'KendallTau Teacher {meters[ema_kentau]:.4f}'.format(
                        epoch, i, len(train_loader), meters=meters))
                log.record(epoch + i / len(train_loader), {
                    'step': global_step,
                    **meters.values(),
                    **meters.averages(),
                    **meters.sums()
                })

        else:
            prec1 = accuracy(class_logit.data, target_var.data, topk=(1,))
            meters.update('top1', prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
            meters.update('error1', 100. - prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())

            ema_prec1 = accuracy(ema_logit.data, target_var.data, topk=(1,))
            meters.update('ema_top1', ema_prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
            meters.update('ema_error1', 100. - ema_prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())

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
                    'Accuracy Student {meters[top1]:.3f}\t'
                    'Accuracy Teacher {meters[ema_top1]:.3f}'.format(
                        epoch, i, len(train_loader), meters=meters))
                log.record(epoch + i / len(train_loader), {
                    'step': global_step,
                    **meters.values(),
                    **meters.averages(),
                    **meters.sums()
                })

        if isinstance(tb_writer, SummaryWriter):
            tb_writer.add_scalar("Loss", loss.cpu().item(), epoch * len(train_loader) + i)
            tb_writer.add_scalar("Class_loss", class_loss.cpu().item(), epoch * len(train_loader) + i)
            if consistency:
                tb_writer.add_scalar("Cons_loss", consistency_loss.cpu().item(), epoch * len(train_loader) + i)
    if isinstance(tb_writer, SummaryWriter):
        if task != 'userid':
            tb_writer.add_scalar("Student_model_training", perfo_score, epoch)
            tb_writer.add_scalar("Teacher_model_training", ema_perfo_score, epoch)
        else:
            tb_writer.add_scalar("Student_model_training", prec1[0].cpu().item(), epoch)
            tb_writer.add_scalar("Teacher_model_training", ema_prec1[0].cpu().item(), epoch)

def validate(eval_loader, model, model_type, log, global_step, epoch, print_freq = 1, device='cuda:0',
             task = 'userid', writer = None):
    """
    Validation function; takes in a model and evaluation dataloader and measure performance of the model
    on the validation dataloader
    :param eval_loader: (DataLoader) dataloader to evaluate
    :param model: (nn.Module) model to evaluate
    :param model_type: (string) student or teacher
    :param log: (TrainLog) log where to print the evaluation performance data
    :param global_step: (int) global learning step currently on
    :param epoch: (int) current epoch during training (record purposes)
    :param print_freq: (int) frequency during which print evaluation performance info
    :param device: (string) device for computations
    :param task: (string) task to perform
    :param writer: (SummaryWriter) tensorboard writer
    :return:
    """
    meters = AverageMeterSet()
    class_criterion = target_criterion_dict[task]

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

        if task == 'userid':
            softmax1 = F.softmax(output1, dim=1)
            class_loss = class_criterion(softmax1, target_var.squeeze())
            writer.add_scalar("{}_valid_loss".format(model_type), class_loss, epoch)
            prec1 = accuracy(softmax1.data, target_var.data, topk=(1,))
            meters.update('class_loss', class_loss.cpu().item(), labeled_minibatch_size.cpu().item())
            meters.update('top1', prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())
            meters.update('error1', 100.0 - prec1[0].cpu().item(), labeled_minibatch_size.cpu().item())

            meters.update('batch_time', time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                LOG.info(
                    'Test: [{0}/{1}]\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Accuracy {meters[top1]:.3f}\t'.format(
                        i, len(eval_loader), meters=meters))
            LOG.info(' * Prec@1 {top1.avg:.3f}\t'
                     .format(top1=meters['top1']))

            log.record(epoch, {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })

            return meters['top1'].avg

        else: # 'pr_mean', 'rt_mean', 'rr_std'
            class_loss = class_criterion(output1, target_var.float())
            writer.add_scalar("{}_valid_loss".format(model_type), class_loss, epoch)
            perfo_score, _ = kendalltau(output1.detach().cpu().numpy(), target_var.detach().cpu().numpy())
            meters.update('class_loss', class_loss.cpu().item(), labeled_minibatch_size.cpu().item())
            meters.update('kentau', perfo_score, labeled_minibatch_size.cpu().item())

            meters.update('batch_time', time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                LOG.info(
                    'Test: [{0}/{1}]\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'KendallTau Score {meters[kentau]:.4f}'.format(
                        i, len(eval_loader), meters=meters))

            log.record(epoch, {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })

            return meters['kentau'].avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    labeled_pred = torch.tensor(([pred[i] for i,ex in enumerate(target) if ex != NO_LABEL]))
    labeled_target = torch.tensor(([target[i] for i, ex in enumerate(target) if ex != NO_LABEL]))

    correct = labeled_pred.eq(labeled_target)

    res = []
    for k in topk:
        correct_k = correct.float().sum()
        res.append(correct_k.mul_(100.0 / float(labeled_minibatch_size)))
    return res

def get_current_consistency_weight(epoch, consistency, consistency_rampup=1000):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def adjust_learning_rate(lr, initial_lr, lr_rampup, lr_rampdown_epochs, optimizer, epoch,
                         epochs, step_in_epoch, total_steps_in_epoch):
    """
    Custom learning adjustment function
    :param lr: (float) maximum learning rate
    :param initial_lr: (float) initial learning rate
    :param lr_rampup: (int) length for lr to reach max value
    :param lr_rampdown_epochs: length for lr to decrease
    :param optimizer: (optimizer) optimizer
    :param epoch: (int) current epoch
    :param epochs: (int) total epochs
    :param step_in_epoch: (int) current batch number)
    :param total_steps_in_epoch: (int) amount of batches in epoch
    :return: new learning rate
    """
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

def create_ema_model(model):
    """
    Exponentially moving average model creation
    :param model: (nn.Module) model on which to deactivate learning
    :return: EMA model
    """
    for param in model.parameters():
        param.detach()
    return model

## Functions copied from original Mean Teacher Repository, https://github.com/CuriousAI/mean-teacher ##

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
