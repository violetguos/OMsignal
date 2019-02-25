'''
Script for training models with explicitly specified hyperparameters.
This can train on the training AND validation set,
in order to have the largest model possible that can be called for test.
(This is okay because the test data is unseen.)
'''

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

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Train models.')
    parser.add_argument('task_name', help='Task to train the model for. Possible choices: [PR, RT, ID]')
    parser.add_argument('--combine', help='Combine train and validation sets.', action='store_true')
    args = parser.parse_args()

    # Make sure arguments are good
    task_names = ['PR', 'RT', 'ID']
    if args.task_name not in task_names:
        raise ValueError('Invalid task name specified. Possible choices are: [PR, RT, ID]')

    # Configure for GPU (or not)
    cluster = torch.cuda.is_available()
    cluster = False
    print('GPU available: {}'.format(cluster))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Import the data, concatenating train and valid sets
    suffix = 'report'
    X_train, X_valid, y_train, y_valid = import_train_valid('all', cluster=cluster)
    if args.combine:
        X_train = np.concatenate((X_train, X_valid), axis=0)
        y_train = np.concatenate((y_train, y_valid), axis=0)
        suffix = 'eval'
    train_batch_size, valid_batch_size = 160, 160

    # Map the id values for the id column
    mapping = get_id_mapping(y_train[:,3])
    y_train[:,3] = map_ids(y_train[:,3], mapping)
    y_valid[:,3] = map_ids(y_valid[:,3], mapping)

    # Preprocess the data (moved back this here since we wont use a dataloader for ranking predictions)
    preprocess = Preprocessor()
    preprocess.to(device)
    X_train = preprocess(torch.from_numpy(X_train)).numpy()
    X_valid = preprocess(torch.from_numpy(X_valid)).numpy()

    # Make augmentations
    trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomNegate(0.5), \
        RandomReplaceNoise(0.5), RandomDropoutBurst(0.5)])
    
    # Train the network
    if args.task_name == 'PR':
        PR_CNN, pr_train_losses, pr_valid_losses = train_PR_CNN(X_train, y_train, X_valid, y_valid,
                                                                train_batch_size, valid_batch_size, device,
                                                                trsfrm=trsfrm, 
                                                                conv1_num_filters=2, conv2_num_filters=2,
                                                                conv_ksize=4, num_linear=128, p=0.8,
                                                                learning_rate=0.1, num_epochs=100)
        torch.save(PR_CNN.state_dict(), os.path.join(MODELS_DIR, 'final_{}_PR_CNN.pt'.format(suffix)))

    elif args.task_name == 'RT':
        RT_Ranker, rt_train_losses, rt_train_accs, \
        rt_valid_losses, rt_valid_accs = train_RT_Ranker(X_train, y_train, X_valid, y_valid,
                                                         train_batch_size, valid_batch_size, device,
                                                         trsfrm=trsfrm, 
                                                         conv1_num_filters=16, conv2_num_filters=16,
                                                         conv_ksize=32, num_linear=256, p=0.8,
                                                         learning_rate=0.1, num_epochs=100)
        torch.save(RT_Ranker.state_dict(), os.path.join(MODELS_DIR, 'final_{}_RT_Ranker.pt'.format(suffix)))

    elif args.task_name == 'ID':
        ID_CNN, id_train_losses, id_train_accs, \
        id_valid_losses, id_valid_accs = train_ID_CNN(X_train, y_train, X_valid, y_valid,
                                                      train_batch_size, valid_batch_size, device,
                                                      trsfrm=trsfrm, 
                                                      conv1_num_filters=16, conv2_num_filters=2,
                                                      conv_ksize=64, num_linear=256, p=0.0,
                                                      learning_rate=0.1, num_epochs=100)
        torch.save(ID_CNN.state_dict(), os.path.join(MODELS_DIR, 'final_{}_ID_CNN.pt'.format(suffix)))