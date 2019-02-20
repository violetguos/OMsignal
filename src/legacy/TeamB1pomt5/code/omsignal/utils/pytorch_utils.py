'''
Functions for training a neural network.
'''
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
import math

from omsignal.utils.dataloader_utils import OM_dataset, Rank_dataset, import_OM, import_train_valid, get_dataloader
from omsignal.utils.scoring_function import scorePerformance
from config import LOG_DIR, MODELS_DIR
from omsignal.base_networks import CNNRegression, CNNClassification
from omsignal.om_networks import CNNRank
from omsignal.utils.preprocessor import Preprocessor
from omsignal.utils.rr_stdev import RR_Regressor, make_prediction
from omsignal.utils.augmentation import RandomCircShift, RandomDropoutBurst, RandomNegate, RandomReplaceNoise
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import scipy.stats as stats
import sklearn.metrics
import warnings
warnings.filterwarnings('ignore')

def get_id_mapping(ids):
    ''' 
    The max value of the ID is 43 and the min value is 0. 
    However, there are only 32 participants - so we map the values down.
    {original_num : mapped_num}
    ''' 
    mapping = {}
    j = 0
    for i in ids:
        if i not in mapping:
            mapping[i] = j
            j += 1
    return mapping

def map_ids(ids, mapping):
    mapped_ids = [mapping[i] for i in ids]
    return np.array(mapped_ids)

def unmap_ids(mapped_ids, mapping):
    reverse_mapping = {v:k for k, v in mapping.items()}
    unmapped_ids = [reverse_mapping[i] for i in mapped_ids]
    return np.array(unmapped_ids)

def train_network(model, task_num, task_type, device, train_dataloader, valid_dataloader, optimizer, loss_func, num_epochs=5, save_name = None):
    if task_type not in ["Regression", "Classification" , "Ranking"]:
        raise ValueError("task_type must be in ['Regression', 'Classification' , 'Ranking']")

    # Set up losses and accuracies
    train_losses, val_losses = [], []
    if task_type != "Regression":
        train_accuracies, val_accuracies = [], []

    min_valid_losses = math.inf # Variable to save save the best model

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))

        # Training
        model.train()
        if task_type != "Regression":
            total, correct = 0, 0
        running_loss = 0.0

        for input, label in tqdm(train_dataloader):
            input, label = input.float().to(device), label[:,task_num].to(device)
            if task_type == "Classification":
                label = label.long()
            else:
                label = label.float()

            # Forward
            outputs = model(input)
            loss = loss_func(outputs, label)
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
            acc = correct/total
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
            input, label = input.float().to(device), label[:,task_num].to(device)
            if task_type == "Classification":
                label = label.long()
            else:
                label = label.float()

            # Forward
            outputs = model(input)
            loss = loss_func(outputs, label)

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
            acc = correct/total
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
    if save_name != None :
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, '{}'.format(save_name))))

    if task_type != "Regression":
        return train_losses, train_accuracies, val_losses, val_accuracies
    else:
        return train_losses, val_losses

def train_PR_CNN(X_train, y_train, X_valid, y_valid, train_batch_size, valid_batch_size, device, trsfrm=None,
                 learning_rate=0.1, num_epochs=10, conv1_num_filters=2, conv2_num_filters=1,
                 conv_ksize=10, conv_stride=1, conv_padding=4,
                 pool_ksize=5, pool_stride=8, pool_padding=1,
                 num_linear=100, p=0.5,
                 log=True):
    task = 0
    train_regression_loader = get_dataloader(X_train, y_train, transform=trsfrm, shuffle=True, batch_size=train_batch_size, task_type="Regression")
    valid_regression_loader = get_dataloader(X_valid, y_valid, transform=None, shuffle=False, batch_size=valid_batch_size, task_type="Regression")

    PR_CNN = CNNRegression(3750, conv1_num_filters=conv1_num_filters, conv2_num_filters=conv2_num_filters,
                           conv_ksize=conv_ksize, conv_stride=conv_stride, conv_padding=conv_padding,
                           pool_ksize=pool_ksize, pool_stride=pool_stride, pool_padding=pool_padding,
                           num_linear=num_linear, p=p)
    PR_CNN.to(device)
    optimizer = torch.optim.Adagrad(PR_CNN.parameters(), weight_decay=0, lr=learning_rate)

    regress_loss_func = nn.MSELoss()
    train_losses, valid_losses = train_network(PR_CNN, task, "Regression", device, 
                                               train_regression_loader, valid_regression_loader, 
                                               optimizer, regress_loss_func, 
                                               num_epochs=num_epochs, save_name="PR_CNN.pt" )
    if log:
        log_training(PR_CNN, task, 'Regression', train_losses, valid_losses)
    return PR_CNN, train_losses, valid_losses

def train_RT_Ranker(X_train, y_train, X_valid, y_valid, train_batch_size, valid_batch_size, device, trsfrm=None,
                    learning_rate=0.1, num_epochs=10, conv1_num_filters=2, conv2_num_filters=1,
                    conv_ksize=10, conv_stride=1, conv_padding=4,
                    pool_ksize=5, pool_stride=8, pool_padding=1,
                    num_linear=100, p=0.5,
                    log=True):
    task = 1
    train_ranking_loader = get_dataloader(X_train, y_train, transform=None, shuffle=True, batch_size=train_batch_size, task_type="Ranking" )
    valid_ranking_loader = get_dataloader(X_valid, y_valid, transform=None, shuffle=False, batch_size=valid_batch_size, task_type="Ranking")
    
    RT_Ranker = CNNRank(3750, conv1_num_filters=conv1_num_filters, conv2_num_filters=conv2_num_filters,
                        conv_ksize=conv_ksize, conv_stride=conv_stride, conv_padding=conv_padding,
                        pool_ksize=pool_ksize, pool_stride=pool_stride, pool_padding=pool_padding,
                        num_linear=num_linear, p=p)
    RT_Ranker.to(device)
    optimizer = torch.optim.Adagrad(RT_Ranker.parameters(), weight_decay=0, lr=learning_rate)
    rank_loss_func = torch.nn.BCEWithLogitsLoss()
    train_losses, train_accs, valid_losses, val_accs = train_network(RT_Ranker, task, "Ranking", device,
                                                                     train_ranking_loader, valid_ranking_loader,
                                                                     optimizer, rank_loss_func, 
                                                                     num_epochs=num_epochs, 
                                                                     save_name="RT_Ranker.pt")
    if log:
        log_training(RT_Ranker, task, 'Ranking', train_losses, valid_losses, 
                     train_accs=train_accs, valid_accs=val_accs)
    return RT_Ranker, train_losses, train_accs, valid_losses, val_accs 
    
def train_ID_CNN(X_train, y_train, X_valid, y_valid, train_batch_size, valid_batch_size, device, trsfrm=None,
                 learning_rate=0.1, num_epochs=10, conv1_num_filters=2, conv2_num_filters=1,
                 conv_ksize=10, conv_stride=1, conv_padding=4,
                 pool_ksize=5, pool_stride=8, pool_padding=1,
                 num_linear=100, p=0.5,
                 log=True):
    task = 3
    train_classification_loader = get_dataloader(X_train, y_train, transform=None, shuffle=True, batch_size=train_batch_size, task_type="Classification")
    valid_classification_loader = get_dataloader(X_valid, y_valid, transform=None, shuffle=False, batch_size=valid_batch_size, task_type="Classification")  

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
                                                                     save_name="ID_CNN.pt")
    if log:
        log_training(ID_CNN, task, 'Classification', train_losses, valid_losses, 
                     train_accs=train_accs, valid_accs=val_accs)
    return ID_CNN, train_losses, train_accs, valid_losses, val_accs

def Predict_and_Score(X, Y, device, PR_model=None, RT_model=None, RR_model="bootleg", ID_model=None):
    # Format PR
    predicted_PR = PR_model(torch.tensor(X).to(device)).cpu().detach().numpy().astype(np.float32).flatten()
    true_PR =  Y[:,0].astype(np.float32).flatten()

    # Format RT
    predicted_RT_rank = RT_model.Predict_ranking(X, device).astype(np.float32).flatten()
    true_RT =  Y[:,1].astype(np.float32).flatten()

    # Predict RR
    predicted_RR = make_prediction(X).astype(np.float32).flatten()
    true_RR =  Y[:,2].astype(np.float32).flatten()

    # Format ID
    predicted_ID = ID_model.Predict_class(X, device).astype(np.int32).flatten()
    true_ID =  Y[:,3].astype(np.int32).flatten()

    overall_score, pr_score, rt_score, rr_score, id_acc = scorePerformance(predicted_PR, true_PR, 
                                                                           predicted_RT_rank, true_RT,
                                                                           predicted_RR, true_RR, 
                                                                           predicted_ID, true_ID)
    return overall_score, pr_score, rt_score, rr_score, id_acc 

def log_training(model, task, task_type, train_losses, valid_losses, train_accs=None, valid_accs=None):
    filename = datetime.datetime.now().isoformat().replace(':', '-').replace('.', '-')
    if task == 0:
        suffix = '_PR'
    elif task == 1:
        suffix = '_RT'
    elif task == 3:
        suffix = '_ID'

    # Log the losses (and accuracies if applicable)
    loss_log = os.path.join(LOG_DIR, '{}{}.losses'.format(filename, suffix))
    if task_type == 'Regression':
        header = 'epoch,train_loss,valid_loss\n'
        with open(loss_log, 'w') as fp:
            fp.write(header)
            for i, train_loss in enumerate(train_losses):
                valid_loss = valid_losses[i]
                fp.write('{},{},{}\n'.format(i, train_loss, valid_loss))
    else:
        header = 'epoch,train_loss,valid_loss,train_acc,valid_acc\n'
        with open(loss_log, 'w') as fp:
            fp.write(header)
            for i, train_loss in enumerate(train_losses):
                valid_loss = valid_losses[i]
                train_acc, valid_acc = train_accs[i], valid_accs[i]
                fp.write('{},{},{},{},{}\n'.format(i, train_loss, valid_loss, train_acc, valid_acc))

    # Log a summary of the model (to keep track of any architecture changes)
    summary_log = os.path.join(LOG_DIR, '{}{}.summary'.format(filename, suffix))
    with open(summary_log, 'w') as fp:
        fp.write(repr(model))
    print('Logged at: {}'.format(filename))

if __name__ == '__main__':
    '''
    Testing for development -
    eventually we should structure proper experiments/grid search
    '''

    # Configure for GPU (or not)
    cluster = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters
    train_batch_size = 16
    valid_batch_size = 160
    hidden_size = 20
    learning_rate = 0.1
    num_epochs = 10

    # Set data-particular vars
    num_classes = 32
    input_size = 3750       # Length of ecg sequence
    fft_input_size = 1876   # Length of fft sequence

    # Import the data
    X_train, X_valid, y_train, y_valid = import_train_valid('all', cluster=cluster)

    # Preprocess the data (moved back this here since we wont use a dataloader for ranking predictions)
    preprocess = Preprocessor()
    preprocess.to(device)
    X_train = preprocess(torch.from_numpy(X_train)).numpy()
    X_valid = preprocess(torch.from_numpy(X_valid)).numpy()

    # Map the id values for the id column
    mapping = get_id_mapping(y_train[:,3])
    y_train[:,3] = map_ids(y_train[:,3], mapping)
    y_valid[:,3] = map_ids(y_valid[:,3], mapping)

    # Make augmentations
    #trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomNegate(0.5), \
    #    RandomReplaceNoise(0.5), RandomDropoutBurst(0.5)])
    trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomDropoutBurst(0.5)])
    
    # Train the various tasks
    print('Training PR task.')
    PR_CNN, pr_train_losses, pr_valid_losses = train_PR_CNN(X_train, y_train, X_valid, y_valid, 
                                                            train_batch_size, valid_batch_size, device, 
                                                            trsfrm=None, learning_rate=0.1, num_epochs=10)

    print('Training RT task.')
    RT_Ranker, rt_train_losses, rt_train_accs, \
    rt_valid_losses, rt_valid_accs = train_RT_Ranker(X_train, y_train, X_valid, y_valid, 
                                                     train_batch_size, valid_batch_size, device, \
                                                     trsfrm=None, learning_rate=0.1, num_epochs=10)

    print('Training ID task.')
    ID_CNN, id_train_losses, id_train_accs, \
    id_valid_losses, id_valid_accs = train_ID_CNN(X_train, y_train, X_valid, y_valid, 
                                                  train_batch_size, valid_batch_size, device, 
                                                  trsfrm=None, learning_rate=0.1, num_epochs=10)

    print(Predict_and_Score(X_valid, y_valid, device, PR_model=PR_CNN, RT_model=RT_Ranker, 
                            RR_model="bootleg", ID_model=ID_CNN))
    
