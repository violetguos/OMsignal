import os
import sys
import torch
import numpy as np
from tensorboardX import SummaryWriter
sys.path.append(os.path.abspath(os.path.join('..', '..')))
import configparser
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.algorithm.ode_multisource_classification import ODEModel
from src.utils.ode_utils import load_data
from src.utils.ode_utils import get_hyperparameters
import torch.nn as nn
import errno 
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
	model.train()

	total, correct = 0, 0
	running_loss = 0.0

	loss_func = nn.CrossEntropyLoss().to(device)

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

# def training_loop(model,optimizer,train_loader,valid_loader,hyperparameters_dict):
# 	if hyperparameters_dict['include_subsamples']:
# 		print('ok')

def training_loop(model,optimizer,train_loader,valid_loader,hyperparameters_dict):


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




def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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
    #main("src/scripts/model_input.in")




# def training_loop(
#         model,
#         optimizer_encoder,
#         optimizer_prediction,
#         criterion,
#         train_loader,
#         unlabeled_loader,
#         eval_loader,
#         score_param_index,
#         hyperparameters_dict,
#         loss_history,
#         chkptg_freq=10,
#         prefix='neural_network',
#         path='./'):
#     # train the model using optimizer / criterion
#     # this function also creates a tensorboard log
#     writer = SummaryWriter(hyperparameters_dict['tbpath'])

#     train_loss_history = []
#     train_acc_history = []
#     valid_loss_history = []
#     valid_acc_history = []
#     weight = hyperparameters_dict['weight']

#     loss_history.prefix = "CNNencoder"
#     loss_history.mode = "train"
#     # Index starts at 1 for reporting purposes
#     for epoch in range(1, hyperparameters_dict['nepoch'] + 1):

#         train_mse_loss = train_unsupervised_per_epoch(
#             model,
#             optimizer_encoder,
#             hyperparameters_dict["batchsize"]*BATCHSIZE_RATIO,
#             unlabeled_loader,
#         )
#         # log the errors everytime!
#         writer.add_scalar('Training/ReconstructLoss', train_mse_loss, epoch)

#         train_loss, train_acc = train_model(
#             model, optimizer_prediction, criterion, train_loader,
#             score_param_index, weight
#         )
#         train_loss_history.append(train_loss)
#         train_acc_history.append(train_acc)

#         valid_loss, valid_acc = eval_model(
#             model, criterion, eval_loader,
#             score_param_index, weight
#         )
#         valid_loss_history.append(valid_loss)
#         valid_acc_history.append(valid_acc)

#         writer.add_scalar('Training/Loss', train_loss, epoch)
#         writer.add_scalar('Valid/Loss', valid_loss, epoch)

#         writer.add_scalar('Training/OverallScore', train_acc[0], epoch)
#         writer.add_scalar('Valid/OverallScore', valid_acc[0], epoch)

#         writer.add_scalar('Training/prMeanTau', train_acc[1], epoch)
#         writer.add_scalar('Valid/prMeanTau', valid_acc[1], epoch)

#         writer.add_scalar('Training/rtMeanTau', train_acc[2], epoch)
#         writer.add_scalar('Valid/rtMeanTau', valid_acc[2], epoch)

#         writer.add_scalar('Training/rrStdDevTau', train_acc[3], epoch)
#         writer.add_scalar('Valid/rrStdDevTau', valid_acc[3], epoch)

#         writer.add_scalar('Training/userIdAcc', train_acc[4], epoch)
#         writer.add_scalar('Valid/userIdAcc', valid_acc[4], epoch)

#         print("Epoch {} {} {} {} {}".format(
#             epoch, train_loss, valid_loss, train_acc, valid_acc)
#         )

#         # Checkpoint
#         if epoch % chkptg_freq == 0:
#             save_model(epoch, model, prefix, path)

#     save_model(hyperparameters_dict['nepoch'], model, prefix, path)
#     return [
#         (train_loss_history, train_acc_history),
#         (valid_loss_history, valid_acc_history)
#     ]