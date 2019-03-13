import argparse
from datetime import datetime
import logging
import configparser
import numpy as np
import torch
import torch.nn as nn

import src.legacy.TABaseline.code.baseline_models as models

from src.scripts.dataloader_utils import import_OM
from src.scripts.mean_teacher.pytorch_utils import task_training, create_ema_model
from src.scripts.mean_teacher.model import Conv1DLinear_Dil
from src.scripts.mean_teacher.data_utils import get_hyperparameters

## Global variables ##

NO_LABEL = 0
args = None
best_prec1 = 0
global global_step
LOG = logging.getLogger('main')

target_criterion_dict = {
    'pr_mean': nn.MSELoss(),
    'rt_mean': nn.MSELoss(),
    'rr_stdev': nn.MSELoss(),
    'userid': nn.CrossEntropyLoss(size_average = True, reduction='sum', ignore_index=NO_LABEL)
}

target_out_size_dict = {
    'pr_mean': 1,
    'rt_mean': 1,
    'rr_stdev': 1,
    'userid': 32,
}


if __name__ == "__main__":

    # Seeding
    np.random.seed(23)

    # Parsing input arguments
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Choose task and model.')
    parser.add_argument('-task', help='Task to train the model for. Possible choices: [pr_mean, rt_mean, rr_stdev, userid]', default='userid')
    parser.add_argument('-model', help='Model to use. dict: [LSTM, RNN, MLP, CONV1D, CONV1DBN, CONV1D_DIL]', default='CONV1D_DIL')
    args = parser.parse_args()

    # Setting up log
    logging.basicConfig(level=logging.INFO)
    global_step = 0

    # Setting up GPU
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device enabled: {}".format(device))

    # Get hyperparameters
    config = configparser.ConfigParser()
    config.sections()
    config.read('src/scripts/mean_teacher/configuration_meanteacher.ini')
    hyperparameters = get_hyperparameters(config)

    normalize = hyperparameters[0]['normalize']

    n_layers = hyperparameters[3]['n_layers']
    hidden_size = hyperparameters[3]['hidden_size']
    kernel_size = hyperparameters[3]['kernel_size']
    pool_size = hyperparameters[3]['pool_size']
    dropout = hyperparameters[3]['dropout']
    dilation = hyperparameters[3]['dilation']

    input_size = 3750
    out_size = target_out_size_dict[args.task]


    # Fetch data -- {}_dataset is a (N_samples, 2) shape vector
    train_data, tr_pr, tr_rt, tr_rr, tr_ids = import_OM("training", cluster=False, normalize_data=normalize)
    train_data = np.concatenate((np.array(train_data),
                                 np.array(tr_pr)[:,np.newaxis],
                                 np.array(tr_rt)[:,np.newaxis],
                                 np.array(tr_rr)[:,np.newaxis],
                                 np.array(tr_ids)[:,np.newaxis]), axis=2)

    valid_data, va_pr, va_rt, va_rr, va_ids = import_OM("validation", cluster=False, normalize_data=normalize)
    valid_data = np.concatenate((np.array(valid_data),
                                 np.array(va_pr)[:,np.newaxis],
                                 np.array(va_rt)[:,np.newaxis],
                                 np.array(va_rr)[:,np.newaxis],
                                 np.array(va_ids)[:,np.newaxis]), axis=2)

    unlabeled_data = import_OM("unlabeled", cluster=False, len=10000, normalize_data=normalize)
    unlabeled_data = np.concatenate((np.array(unlabeled_data)[:,np.newaxis,:],
                                     np.float(-1) *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis],
                                     np.float(-1) *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis],
                                     np.float(-1) *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis],
                                     -1 *np.ones(len(unlabeled_data))[:,np.newaxis,np.newaxis]), axis=2)

    # Define and Instanciate the model
    for hidden_size in [24,32]:
        for kernel_size in [4,5,6]:
            tb_path = hyperparameters[5]['tbpath'] + "/{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.now()) + "/hs-{}_ks-{}_di-{}_ps-{}".format(hidden_size, kernel_size, dilation, pool_size)
            model = Conv1DLinear_Dil(
                1, out_size, hidden_size, kernel_size, pool_size)
            model_ema = Conv1DLinear_Dil(
                1, out_size, hidden_size, kernel_size, pool_size)
            model_ema = create_ema_model(model_ema)

            # Model to gpu, Create optimizer, criterion and train
            model.to(device)
            model_ema.to(device)

            evaluation_epochs = 1
            checkpoint_epochs = 20


            task_training(args.task, model, model_ema, train_data, valid_data, unlabeled_data, device=device,
                          **hyperparameters[1], **hyperparameters[2], **hyperparameters[4], tbpath=tb_path)