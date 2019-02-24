import numpy as np
import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import get_dataloader

from src.legacy.TeamB1pomt5.code.config import LOG_DIR, MODELS_DIR
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import import_train_valid
from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNClassification
from src.scripts.dataloader_utils import import_OM
from src.scripts.pytorch_utils import train_network
from torch.utils.data import DataLoader, TensorDataset
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import log_training
from tqdm import tqdm

"""
Class 2 naive implementation
- Entropy minimization term added

The naive implementation of our class 2 model is to have a "mega" loop, doing repeatedly
a dataloader creation, training process and unlabeled data evaluation. The first iteration
we use only original labeled data to train a prediction model. We then evaluate the performance
of this new model on the unlabeled data. Data with sufficiently high confidence in their prediction are considered "safe"
and used as labeled data for the next training iteration (integrated to new train loader)

-for this prototype, ignore tasks other than ID classif.
"""
def merge_into_training(train_data, train_label, new_labeled_data, new_train_label, shuffle = True):
    """

    :param train_data: n samples x 1 channel x 3751 dimensions, previously used training dataset
    :param new_labeled_data: k samples x 1 channel x 3751 dimensions newly labeled training data
    :return:
    """
    train_data = np.concatenate((train_data, new_labeled_data), axis=0)
    train_label = np.concatenate((train_label, new_train_label), axis=0)

    return train_data, train_label

def fetch_dataloaders(train_data, train_labels, valid_data, valid_labels, batch_size=50, transform=None): #No need with train_ID_CNN
    """
    fetch_dataloders is a function which creates the dataloaders required for data training
    :param train_data: n samples x 1 channel x m dimensions (3750 + 4/1)
    :param valid_data:
    :param unlabeled_data:
    :return: dataloaders of input data
    """
    train_loader = get_dataloader(train_data, train_labels, transform, batch_size=batch_size, task_type="Regression")
    valid_loader = get_dataloader(valid_data, valid_labels, transform, batch_size=batch_size, task_type="Regression") #Only uses original validation data

    return train_loader, valid_loader

def training_loop(training_dataloader, validation_dataloader, model):
    """

    :param training_dataloader:
    :param validation_dataloader:
    :param model:
    :return: model
    """

def evaluate_unlabeled(unlabeled_data, threshold=0.5):
    """
    Unlabeled data is not shufled when turned into tensor for evaluation
    Making it easier to give it back to training at beginning of loop
    :param unlabeled_data: data to be predicted
    :param threshold: confidence threshold for accepting label as true
    :return: new_labeled_data
    """


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Train models.')
    # parser.add_argument('task_name', help='Task to train the model for. Possible choices: [PR, RT, ID]')
    # parser.add_argument('--combine', help='Combine train and validation sets.', action='store_true')
    args = parser.parse_args()

    # Seeding
    np.random.seed(23)

    # Configure for GPU (or not)
    cluster = torch.cuda.is_available()
    # cluster = False
    print('GPU available: {}'.format(cluster))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Import the data but only ID labels, concatenating train and valid sets
    X_train, X_valid, y_train, y_valid = import_train_valid('ids', cluster=cluster)
    # if args.combine:
    #     X_train = np.concatenate((X_train, X_valid), axis=0)
    #     y_train = np.concatenate((y_train, y_valid), axis=0)
    train_batch_size, valid_batch_size = 160, 160

    #Import unlabeled data
    unlabeled = import_OM("unlabeled")
    unlabeled = unlabeled[:,np.newaxis,:]

    #Defining ID Classification model
    model = CNNClassification(3750, 32, conv1_num_filters=16, conv2_num_filters=2, conv_ksize=64, conv_stride=1, conv_padding=4,
                                               pool_ksize=5, pool_stride=8, pool_padding=1,  num_linear=100, p=0.5)
    model.to(device)
    #Defining optimizer
    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=0, lr=0.1)

    epochs = 5
    loss_function = torch.nn.NLLLoss()

    temp_flag = True
    new_data = []
    new_labels = []

    while (temp_flag):

        # Incorporating new samples into training array
        # new_train_data, new_train_label = merge_into_training(X_train, y_train, new_data, new_labels)

        # Creating dataloaders
        # train_loader, valid_loader = fetch_dataloaders(new_train_data, new_train_label, X_valid, y_valid)
        train_loader, valid_loader = fetch_dataloaders(X_train, y_train, X_valid, y_valid)
        print(type(train_loader), '\n')
        for sample, label in train_loader:
            print(sample, np.shape(sample), label, np.shape(label), np.amax(label, axis=0))
            break
        # Training
        train_losses, train_accs, valid_losses, val_accs = train_network(model, 0, "Classification", device, train_loader,
                                                                         valid_loader, optimizer, loss_function,
                                                                         save_name="TestModel", num_epochs=epochs, entropy=False)
        log_training(model, 3, 'Classification', train_losses, valid_losses,
                     train_accs=train_accs, valid_accs=val_accs)

        #new_data, new_labels = evaluate_unlabeled(torch.Tensor(unlabeled))

        temp_flag = False

