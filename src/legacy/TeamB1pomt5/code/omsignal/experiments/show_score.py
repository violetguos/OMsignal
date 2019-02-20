'''
Script for getting the final score of all our best models together.
'''
import numpy as np
import math
import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from config import LOG_DIR, MODELS_DIR
from omsignal.utils.dataloader_utils import import_train_valid
from omsignal.utils.pytorch_utils import get_id_mapping, map_ids, Predict_and_Score
from omsignal.utils.preprocessor import Preprocessor
from omsignal.utils.augmentation import RandomCircShift, RandomDropoutBurst, RandomNegate, RandomReplaceNoise
from omsignal.base_networks import CNNRegression, CNNClassification
from omsignal.om_networks import CNNRank
from torchvision import transforms
from omsignal.utils.rr_stdev import RR_Regressor



if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Show and log the scores from trained models.')
    parser.add_argument('--combine', help='Combine train and validation sets.', action='store_true')
    args = parser.parse_args()
    tag = 'eval' if args.combine else 'report'

    # Configure for GPU (or not)
    cluster = torch.cuda.is_available()
    print('GPU available: {}'.format(cluster))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    location = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Import the data
    X_train, X_valid, y_train, y_valid = import_train_valid('all', cluster=cluster)

    # Preprocess the data
    preprocess = Preprocessor()
    preprocess.to(device)
    X_train = preprocess(torch.from_numpy(X_train)).numpy()
    X_valid = preprocess(torch.from_numpy(X_valid)).numpy()

    # Map the id values for the id column
    mapping = get_id_mapping(y_train[:,3])
    y_train[:,3] = map_ids(y_train[:,3], mapping)
    y_valid[:,3] = map_ids(y_valid[:,3], mapping)

    # Make augmentations
    trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomNegate(0.5), \
        RandomReplaceNoise(0.5), RandomDropoutBurst(0.5)])

    # Load best trained models with best hyperparameters
    PR_CNN = CNNRegression(3750, conv1_num_filters=2, conv2_num_filters=2,
                           conv_ksize=4, num_linear=128, p=0.8)
    PR_CNN.to(device)
    PR_CNN.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'final_{}_PR_CNN.pt'.format(tag)), 
                           map_location=location))
    PR_CNN.eval()
    

    RT_Ranker = CNNRank(3750, conv1_num_filters=16, conv2_num_filters=16,
                        conv_ksize=32, num_linear=256, p=0.8)
    RT_Ranker.to(device)
    RT_Ranker.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'final_{}_RT_Ranker.pt'.format(tag)),
                              map_location=location))
    RT_Ranker.eval()
    

    ID_CNN = CNNClassification(1876, 32,
                               conv1_num_filters=16, conv2_num_filters=2,
                               conv_ksize=64, num_linear=256, p=0.0, conv_stride=1, conv_padding=4,
                               pool_ksize=5, pool_stride=8, pool_padding=1)
    ID_CNN.to(device)
    ID_CNN.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'final_{}_ID_CNN.pt'.format(tag)),
                           map_location=location))
    ID_CNN.eval()

    # Get training and validation score using all models
    print('Training scores:')
    print('Overall, PR, RT, RR, ID')
    train_scores = Predict_and_Score(X_train, y_train, device, PR_model=PR_CNN, RT_model=RT_Ranker, 
                                     RR_model="bootleg", ID_model=ID_CNN)
    print(train_scores)

    print()

    print('Validation scores:')
    print('Overall, PR, RT, RR, ID')
    valid_scores = Predict_and_Score(X_valid, y_valid, device, PR_model=PR_CNN, RT_model=RT_Ranker, 
                                     RR_model="bootleg", ID_model=ID_CNN)
    print(valid_scores)

    # Log the scores
    filename = '{}_scores.log'.format(tag)
    log_path = os.path.join(LOG_DIR, filename)
    with open(log_path, 'w') as fp:
        fp.write(tag)
        fp.write('\n')
        fp.write('Train: {}'.format(train_scores))
        fp.write('\n')
        fp.write('Valid: {}'.format(valid_scores))