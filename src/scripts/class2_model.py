import numpy as np
import argparse
import math
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from src.legacy.TeamB1pomt5.code.config import LOG_DIR, MODELS_DIR
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import import_train_valid
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import get_id_mapping, map_ids, Predict_and_Score, train_PR_CNN, train_RT_Ranker, train_ID_CNN
from src.legacy.TeamB1pomt5.code.omsignal.utils.augmentation import RandomCircShift, RandomDropoutBurst, RandomNegate, RandomReplaceNoise
from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNRegression, CNNClassification
from src.legacy.TeamB1pomt5.code.omsignal.om_networks import CNNRank
from torchvision import transforms
from src.scripts.dataloader_utils import import_OM

"""
Class 2 naive implementation
- Entropy minimization term
"""

def training_loop(training_dataloader, validation_dataloader, model):
    """

    :param training_dataloader:
    :param validation_dataloader:
    :param unlabeled_dataloader:
    :param model:
    :return:
    """

def evaluate_unlabeled(unlabeled_dataloader, threshold):
    """

    :param unlabeled_dataloader: data to be predicted
    :param threshold: confidence threshold for accepting label as true
    :return: new_labeled_data
    """


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Train models.')
    # parser.add_argument('task_name', help='Task to train the model for. Possible choices: [PR, RT, ID]')
    # parser.add_argument('--combine', help='Combine train and validation sets.', action='store_true')
    args = parser.parse_args()

    # Configure for GPU (or not)
    cluster = torch.cuda.is_available()
    # cluster = False
    print('GPU available: {}'.format(cluster))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Import the data, concatenating train and valid sets
    # suffix = 'report'
    X_train, X_valid, y_train, y_valid = import_train_valid('all', cluster=cluster)
    if args.combine:
        X_train = np.concatenate((X_train, X_valid), axis=0)
        y_train = np.concatenate((y_train, y_valid), axis=0)
        #suffix = 'eval'
    train_batch_size, valid_batch_size = 160, 160

    #Import unlabeled data
    unlabeled = import_OM("unlabeled")

    # Entering Training mega-loop

    # Step 1 : Defining dataloaders