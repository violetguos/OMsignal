import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
sys.path.append(os.path.abspath(os.path.join('..', '..')))
import configparser
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.algorithm.ode_multisource_classification import ODEModel
from src.utils.ode_utils import (
	load_data,
	get_hyperparameters,
	mkdir_p
	)


# BEGIN Global variables #
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
preprocess = Preprocessor()
preprocess.to(device)

# fix  seed for reproducibility
seed = 54
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def train_epoch(model, optimizer, loader,epoch,include_subsamples,large):
	"""
    Trains model for a single epoch
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param loader: the training set loader
    :param include_subsamples: whether to train the principal ode_network with sub samples of the signal
    :param large: whether this epoch trains a full signal or subsample
    
    :return: training loss, accuracy, large (for next epoch)
    """
	total, correct = 0, 0
	running_loss = 0.0

	loss_func = nn.CrossEntropyLoss().to(device)
	model.train()

	for i, (data, label) in enumerate(loader):
	    data, label = data.float().to(device), label[:, 0].to(device)
	    
	    label = label.long()

	    if include_subsamples:
	        if not large:

	            if epoch%5==0:
	                data = data[:,:,:750]
	                if large:
	                    large = False
	                else:
	                    large = True
	            elif epoch%4 ==0:
	                data = data[:,:,3000:]
	            elif epoch%3==0:
	                data = data[:,:,2250:3000]
	            elif epoch%2==0:
	                data = data[:,:,1500:2250]
	            elif epoch%1==0:
	                data = data[:,:,750:1500]
	        
	    X = preprocess(data.cpu()).numpy()
	    fft = torch.Tensor(np.fft.rfft(X, axis = 2).astype(np.float32)).to(device)
	    outputs = model(data,fft)
	    optimizer.zero_grad()
	    
	    loss = loss_func(outputs, label)

	    # Backward
	    loss.backward()
	    
	    # Update
	    optimizer.step()


	    _, predicted = torch.max(outputs.data, 1)
	    correct += (predicted == label).sum().item()
	    total += label.size(0)
	    
	    running_loss += loss.item()
	    
	t_loss = running_loss;

	acc_statement = ''
	acc = correct / total
	t_acc = acc;

	return t_loss,t_acc,large


def eval_epoch(model, optimizer, loader,max_valid_acc,hyperparameters_dict):
	"""
     Evaluates model for a single epoch
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param loader: the validation set loader
    :param max_valid_acc: stored to save model with highest validation accuracy so far
    :param hyperparameters_dict: stores save location of model
    
    :return: validation loss, validation accuracy, current max validation accuracy
    """
	loss_func = nn.CrossEntropyLoss().to(device)

	model.eval()

	total, correct = 0, 0
	running_loss = 0.0

	for i, (data, label) in enumerate(loader):
	    
	    data, label = data.float().to(device), label[:, 0].to(device) 
	    label = label.long()

	    X = preprocess(data.cpu()).numpy()
	    fft = torch.Tensor(np.fft.rfft(X, axis = 2).astype(np.float32)).to(device)
	    outputs = model(data,fft)        
	    loss = loss_func(outputs, label)
	    _, predicted = torch.max(outputs.data, 1)
	    correct += (predicted == label).sum().item()
	    total += label.size(0)  
	    running_loss += loss.item()

	v_loss = running_loss; 
	acc = correct / total
	v_acc = acc

	if acc >= max_valid_acc:
	    max_valid_acc = acc
	    print('validation_best:',acc)
	    torch.save(model.state_dict(), os.path.join(hyperparameters_dict['modelpath'], '{}'.format('best_validation')))

	return v_loss, v_acc, max_valid_acc

def training_loop(model,optimizer,train_loader,valid_loader,hyperparameters_dict):
	"""
     Runs training loop
    :param model: the model created under src/algorithms
    :param optimizer: pytorch optim
    :param train_loader: the training set loader
    :param valid_loader: the validation set loader
    :param hyperparameters_dict: stores save location of model
    
    :return: [(train_losses,train_accuracy)(valid_losses,valid_accuracy)]
    """

	writer = SummaryWriter(hyperparameters_dict['tbpath'])
	include_subsamples = hyperparameters_dict['subsample']
	large = True
	max_valid_acc = 0


	train_loss_history = []
	train_acc_history = []
	valid_loss_history = []
	valid_acc_history = []

	prefix = "ODE_Solver_class"
	# Index starts at 1 for reporting purposes
	for epoch in range(1, hyperparameters_dict['nepoch'] + 1):

	    train_loss, train_acc,large = train_epoch(
	        model, optimizer, train_loader,epoch,include_subsamples,large
	    )

	    train_loss_history.append(train_loss)
	    train_acc_history.append(train_acc)

	    valid_loss, valid_acc,max_valid_acc = eval_epoch(
	       model, optimizer, valid_loader,max_valid_acc,hyperparameters_dict
	    )
	    valid_loss_history.append(valid_loss)
	    valid_acc_history.append(valid_acc)


	    writer.add_scalar('Training/Loss', train_loss, epoch)
	    writer.add_scalar('Valid/Loss', valid_loss, epoch)

	    writer.add_scalar('Training/Acc', train_acc, epoch)
	    writer.add_scalar('Valid/Acc', valid_acc, epoch)

	    print("Epoch {} {} {} {} {}".format(
	        epoch, train_loss, valid_loss, train_acc, valid_acc)
	    )

	torch.save(model.state_dict(), os.path.join(hyperparameters_dict['modelpath'], '{}'.format('final_model')))

	return [
	    (train_loss_history, train_acc_history),
	    (valid_loss_history, valid_acc_history)
	]


def run(model_hp_dict):

	train_loader, valid_loader = load_data(model_hp_dict,use_gpu,device)

	model = ODEModel()
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(),lr=model_hp_dict["learning_rate"], betas=(.5, .999), weight_decay=0)

	training_loop(model,optimizer,train_loader,valid_loader,model_hp_dict)


def main(config_model):

    # reads in the config files
    model_config = configparser.ConfigParser()
    model_config.read(config_model)
    model_hp_dict = get_hyperparameters(model_config)
    mkdir_p(model_hp_dict["modelpath"])
    # Call functions to run models
    run(model_hp_dict)


if __name__ == "__main__":

    # Read the ini file name from sys arg to avoid different people's different local set up
    # Use a shell script instead to run on your setup

    main(sys.argv[1])