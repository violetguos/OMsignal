# ECG-processing

IFT6759 Winter 2019
OM Signal Project Block 1
Baseline codebase

Authors:
Arsene Fansi Tchango
Simon Blackburn

Usage:

baseline_multitask_main.py : trains a model specified by the parameters in an input file

python base_multitask_main.py input.in 

The code logs training information in TensorboardX (path defined in the input file) and saves the models every 50 epochs.
This code supports multitask learning (3 regression + 1 classification). 
The weight of each loss (MSE / CrossEntropy) can be weighted according to LOSS_WEIGHT as defined in the input file. 
The code supports 5 models:
1) MLP: a simple multi-layer perceptron with 1 hidden layer
2) RNN: recurrent neural network considering the data as an ordered series
3) LSTM: long short term memory that should have better behavior than RNN for long-term dependencies
4) CNN1D: a 1-layer CNN for 1D signal
5) CNN1DBN: a multi-layer CNN with batch-normalization for 1D signal

DataLoaders are created for the training and validation datasets.
This relies on ecgdataset, as defined in the ecgdataset.py file.
We support any combination of targets, as defined by the kwarg targets.

The model is then defined as per the relevant arguments.
The optimized is Adam by default. This can be changed.
The criterion are defined: MSELoss() for regression tasks, CrossEntropyLoss() for classification.
Loss is defined as the sum of the loss for each task, weighted by LOSS_WEIGHT (defined by a list as hyperparemeters).We then call the training_loop function.

Training_loop:
This loop iterates for a fixed number of epochs (as defined by kwarg NEPOCH).
Accuracy / loss are tracked in a tensorboard file with path defined by kwarg TB_PATH
The model is saved every chkptg_freq epochs.
At each epoch, training is done by the train_model function.

train_model:
this function iterates over mini-batches, and changes the model parameters (via backprop).



baseline_models.py : contains the models used for baseline

data_augmentation.py : data augmentation methods. 

preprocessor.py : from OMSignal, it shifts the (moving) average to 0 and reduces the noise in the signal

