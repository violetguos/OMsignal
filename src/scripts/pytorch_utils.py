import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
import math

from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import get_dataloader
from src.legacy.TeamB1pomt5.code.config import MODELS_DIR
from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNClassification
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import log_training
from tqdm import tqdm
from torch.nn.functional import log_softmax

import warnings
warnings.filterwarnings('ignore')



def train_ID_CNN(X_train, y_train, X_valid, y_valid, train_batch_size, valid_batch_size, device, trsfrm=None,
                 learning_rate=0.1, num_epochs=10, conv1_num_filters=2, conv2_num_filters=1,
                 conv_ksize=10, conv_stride=1, conv_padding=4,
                 pool_ksize=5, pool_stride=8, pool_padding=1,
                 num_linear=100, p=0.5,
                 log=True, entropy = True):
    task = 3
    train_classification_loader = get_dataloader(X_train, y_train, transform=None, shuffle=True,
                                                 batch_size=train_batch_size, task_type="Classification")
    valid_classification_loader = get_dataloader(X_valid, y_valid, transform=None, shuffle=False,
                                                 batch_size=valid_batch_size, task_type="Classification")

    ID_CNN = CNNClassification(1876, 32,
                               conv1_num_filters=conv1_num_filters, conv2_num_filters=conv2_num_filters,
                               conv_ksize=conv_ksize, conv_stride=conv_stride, conv_padding=conv_padding,
                               pool_ksize=pool_ksize, pool_stride=pool_stride, pool_padding=pool_padding,
                               num_linear=num_linear, p=p)
    ID_CNN.to(device)
    print(repr(ID_CNN))
    optimizer = torch.optim.Adagrad(ID_CNN.parameters(), weight_decay=0, lr=learning_rate)
    classify_loss_func = nn.NLLLoss()
    train_losses, train_accs, valid_losses, val_accs = train_network(ID_CNN, task, "Classification",
                                                                     device, train_classification_loader,
                                                                     valid_classification_loader, optimizer,
                                                                     classify_loss_func,
                                                                     num_epochs=num_epochs,
                                                                     save_name="ID_CNN.pt", entropy = entropy)
    if log:
        log_training(ID_CNN, task, 'Classification', train_losses, valid_losses,
                     train_accs=train_accs, valid_accs=val_accs)
    return ID_CNN, train_losses, train_accs, valid_losses, val_accs


def train_network(model, task_num, task_type, device, train_dataloader, valid_dataloader, optimizer, loss_func, entropy = True,
                  num_epochs=5, save_name=None):
    if task_type not in ["Regression", "Classification", "Ranking"]:
        raise ValueError("task_type must be in ['Regression', 'Classification' , 'Ranking']")

    # Set up losses and accuracies
    train_losses, val_losses = [], []
    if task_type != "Regression":
        train_accuracies, val_accuracies = [], []

    min_valid_losses = math.inf  # Variable to save save the best model

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))

        # Training
        model.train()
        if task_type != "Regression":
            total, correct = 0, 0
        running_loss = 0.0

        for input, label in tqdm(train_dataloader):
            input, label = input.float().to(device), label[:, task_num].to(device)
            if task_type == "Classification":
                label = label.long()
            else:
                label = label.float()

            # Forward
            outputs = model(input)
            loss = loss_func(outputs, label)
            if entropy:
                loss += entropy_term(outputs)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

            if task_type == "Classification":
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
            if task_type == "Ranking":
                correct += (torch.eq(torch.gt(torch.sigmoid(outputs), 0.5), label.byte())).sum().item()
                total += label.size(0)
            running_loss += loss.item()

        acc_statement = ''
        if task_type != "Regression":
            acc = correct / total
            train_accuracies.append(acc)
            acc_statement = '\t Accuracy: {0:.2f}'.format(acc)
        train_losses.append(running_loss)
        train_statement = 'Train: \t Loss: {0:.2f} {1}'.format(running_loss, acc_statement)
        print(train_statement)

        # Validation
        model.eval()
        if task_type != "Regression":
            total, correct = 0, 0
        running_loss = 0.0

        for input, label in valid_dataloader:
            input, label = input.float().to(device), label[:, task_num].to(device)
            if task_type == "Classification":
                label = label.long()
            else:
                label = label.float()

            # Forward
            outputs = model(input)
            loss = loss_func(outputs, label)
            if entropy:
                loss += entropy_term(outputs)

            if task_type == "Classification":
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
            if task_type == "Ranking":
                correct += (torch.eq(torch.gt(torch.sigmoid(outputs), 0.5), label.byte())).sum().item()
                total += label.size(0)
            running_loss += loss.item()

        acc_statement = ''
        if task_type != "Regression":
            acc = correct / total
            val_accuracies.append(acc)
            acc_statement = '\t Accuracy: {0:.2f}'.format(acc)

        if running_loss < min_valid_losses:
            min_valid_losses = running_loss
            if save_name != None:
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, '{}'.format(save_name)))

        val_losses.append(running_loss)
        val_statement = 'Valid: \t Loss: {0:.2f} {1}'.format(running_loss, acc_statement)
        print(val_statement)

    # Reload the best model we saved
    if save_name != None:
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, '{}'.format(save_name))))

    if task_type != "Regression":
        return train_losses, train_accuracies, val_losses, val_accuracies
    else:
        return train_losses, val_losses


def entropy_term(outputs):
    """
    Function for computing the entropy term of the loss function
    :param outputs: outputs on which to compute the NLL
    :return: loss term
    """
    loss = -1. * torch.mean(torch.sum((outputs * torch.exp(outputs)), dim=1), dim=0)
    loss = torch.autograd.Variable(loss, requires_grad=True)
    return loss