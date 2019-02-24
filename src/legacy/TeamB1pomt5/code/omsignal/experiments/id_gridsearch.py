''' 
Functions for gridsearch for hyperparameter tuning.
'''
import numpy as np
import math
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from config import LOG_DIR, MODELS_DIR
from omsignal.utils.dataloader_utils import import_train_valid
from omsignal.utils.preprocessor import Preprocessor
from omsignal.utils.pytorch_utils import get_id_mapping, get_dataloader, map_ids, train_ID_CNN
from omsignal.utils.augmentation import RandomCircShift, RandomDropoutBurst, RandomNegate, RandomReplaceNoise

from sklearn.model_selection import ParameterGrid
from torchvision import transforms

if __name__ == '__main__':
    # Configure for GPU (or not)
    cluster = torch.cuda.is_available()
    cluster = False
    print('GPU available: {}'.format(cluster))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set data-particular vars
    num_classes = 32
    input_size = 3750       # Length of ecg sequence
    fft_input_size = 1876   # Length of fft sequence
    train_batch_size = 160
    valid_batch_size = 160  # Since we need to predict order of full sequence (all 160)

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
    #trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomDropoutBurst(0.5)])

    

    # Set hyperparameter ranges to test over
    param_grid = {'learning_rate': [0.01, 0.1],
                  'num_epochs': [100],
                  'conv1_num_filters': [2, 16],
                  'conv2_num_filters': [2, 16],
                  'conv_ksize': [4, 32, 64],
                  'num_linear': [16, 128, 256],
                  'dropout_p': [0.0, 0.5, 0.8]
                  }


    # Perform gridsearch - we want to keep params with the lowest validation loss.
    current_min = math.inf
    best_config, best_model = None, None
    grid = ParameterGrid(param_grid)
    for params in grid:
        try:
            ID_CNN, _, id_valid_losses, _, _ = train_ID_CNN(X_train, y_train, X_valid, y_valid, 
                                                            train_batch_size, valid_batch_size, device, 
                                                            trsfrm=trsfrm, 
                                                            learning_rate=params['learning_rate'], 
                                                            num_epochs=params['num_epochs'],

                                                            conv1_num_filters=params['conv1_num_filters'],
                                                            conv2_num_filters=params['conv2_num_filters'],
                                                            conv_ksize=params['conv_ksize'],
                                                            num_linear=params['num_linear'],
                                                            p=params['dropout_p']
                                                            )
        except Exception as e:
            # This hyperparameter configuration is impossible - 
            # e.g. padding bigger than the kernel size, etc. Just skip it.
            print('This configuration is impossible.')
            continue

        # Our training functions already save the best model according to this criterion,
        # so we can just take the minimum of the returned validation losses for each
        # hyperparameter config and see which config gives the lowest of these.
        min_val_loss = min(id_valid_losses)
        if min_val_loss < current_min:
            current_min = min_val_loss
            best_config = params
            best_model = ID_CNN

    # Save the best model explicitly so we don't have to track it down
    print('Done. The config giving the lowest validation loss is:')
    print(best_config)
    print('Which gives loss: {}'.format(current_min))
    torch.save(best_model.state_dict(), os.path.join(MODELS_DIR, 'best_ID_CNN.pt'))

    # Log this info - just in case retraining is needed by hand
    with open(os.path.join(LOG_DIR, 'ID_best_hyperparams.txt'), 'w') as fp:
        fp.write(str(best_config))
        fp.write('\n')
        fp.write('Lowest validation loss: {}'.format(current_min))