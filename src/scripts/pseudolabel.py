import numpy as np
import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import get_dataloader

from src.legacy.TeamB1pomt5.code.config import LOG_DIR, MODELS_DIR
from src.scripts.dataloader_utils import import_train_valid
from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNClassification
from src.scripts.dataloader_utils import import_OM
from src.scripts.pytorch_utils import train_network
from src.legacy.TeamB1pomt5.code.omsignal.utils.augmentation import RandomCircShift, RandomDropoutBurst, RandomNegate, RandomReplaceNoise
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import log_training, get_id_mapping, map_ids
from tqdm import tqdm
from torchvision import transforms


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

def evaluate_unlabeled(unlabeled_data, model, device, threshold=0.8):
    """
    This function takes in a vector of unlabeled_data, a model and a threshold.
    It predicts a label for the unlabeled data and keeps the prediction with
    sufficiently high confidence
    :param unlabeled_data: data to be predicted
    :param model: prediction model
    :param threshold: confidence threshold for accepting label as true
    :return: labeled, unlabeled, top_pred: returns a np array of the labeled data with label,
    a numpy array of still unlabeled data and an array describing the labeled data
    """
    best_preds = []
    best_preds_label = []
    best_samples = []
    pred = []
    labeled = []
    unlabeled = []
    labeled_idx = []
    # unlabeled_data = unlabeled_data.reshape(unlabeled_data.shape[0], 1, unlabeled_data.shape[1]).float().cuda()
    unlabeled_data = unlabeled_data.astype(np.float32)

    output = model(torch.Tensor(unlabeled_data).cuda()) # Really need to store that?
    sum__ = torch.exp(output[0])
    sum_ = torch.sum(torch.exp(output[0]))
    pred_prob= torch.max(torch.exp(output.data), 1)[0]
    pred_class = torch.max(torch.exp(output.data), 1)[1]
    print(output, output.data)
    best_preds = [pred_prob[i] for i in pred_prob if i > threshold]
    best_preds_label = [pred_class[i] for i in pred_prob if i > threshold]
    best_samples = [unlabeled_data[i] for i in pred_prob if i > threshold]
    best_idx = [i for i in pred_prob if i > threshold]
        # labeled.append(sample[0].data)
        # pred.append(pred_class[0])
        # best_preds.append((idx, pred_prob[0], pred_class[0]))
        # labeled_idx.append(idx)

    unlabeled = [unlabeled_data[i] for i in range(1, len(unlabeled_data)) if i not in best_idx]

    # print(np.shape(np.array(labeled)[:,np.newaxis]), np.shape(np.array(pred)[:,np.newaxis]), np.shape(unlabeled))
    # return np.concatenate((np.array(labeled)[:,np.newaxis],np.array(pred)[:,np.newaxis]),axis=1), np.array(unlabeled), best_preds
    return np.concatenate((np.array(best_samples[:,np.newaxis]), np.array(best_preds_label[:,np.newaxis])), dim=1), np.array(unlabeled)

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Train models.')
    # parser.add_argument('task_name', help='Task to train the model for. Possible choices: [PR, RT, ID]')
    # parser.add_argument('--combine', help='Combine train and validation sets.', action='store_true')
    args = parser.parse_args()

    # Seeding
    np.random.seed(23)

    torch.cuda.init()

    # Configure for GPU (or not)
    # cluster = torch.cuda.is_available()
    cluster = False
    print('On cluster: {}'.format(cluster))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('GPU available: {}'.format(device))

    # Import the data but only ID labels, concatenating train and valid sets
    X_train, X_valid, y_train, y_valid = import_train_valid('ids', cluster=cluster)

    # if args.combine:
    #     X_train = np.concatenate((X_train, X_valid), axis=0)
    #     y_train = np.concatenate((y_train, y_valid), axis=0)
    train_batch_size, valid_batch_size = 160, 160

    # Remapping IDs
    mapping = get_id_mapping(y_train[:,0])
    y_train[:,0] = map_ids(y_train[:,0], mapping)
    y_valid[:,0] = map_ids(y_valid[:,0], mapping)

    # Preprocess the data (moved back this here since we wont use a dataloader for ranking predictions)
    preprocess = Preprocessor()
    preprocess.to(device)
    X_train = preprocess(torch.from_numpy(X_train)).numpy()
    X_valid = preprocess(torch.from_numpy(X_valid)).numpy()

    #Import unlabeled data
    unlabeled = import_OM("unlabeled", cluster=cluster)
    unlabeled = unlabeled[:,np.newaxis,:].astype(np.float32)
    unlabeled = preprocess(torch.from_numpy(unlabeled)).numpy()

    # Add transformations
    trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomNegate(0.5), \
        RandomReplaceNoise(0.5), RandomDropoutBurst(0.5)])

    #Defining ID Classification model
    model = CNNClassification(3750, 32, conv1_num_filters=32, conv2_num_filters=32, conv_ksize=64, conv_stride=1, conv_padding=4,
                                               pool_ksize=5, pool_stride=8, pool_padding=1,  num_linear=1000, p=0.5)
    model.to(device)

    #Defining optimizer
    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=1E-3, lr=0.005)

    epochs = 70
    loss_function = torch.nn.NLLLoss()

    i = 0
    while i < 3:

        # Creating dataloaders
        train_loader, valid_loader = fetch_dataloaders(X_train, y_train, X_valid, y_valid, transform=trsfrm)

        # train_loader, valid_loader = fetch_dataloaders(X_train, y_train, X_valid, y_valid, transform=trsfrm)

        # Training
        model.train()
        train_losses, train_accs, valid_losses, val_accs = train_network(model, 0, "Classification", device, train_loader,
                                                                         valid_loader, optimizer, loss_function,
                                                                         save_name="TestModel", num_epochs=epochs, entropy=True)

        # Logging training info
        log_training(model, 3, 'Classification', train_losses, valid_losses,
                     train_accs=train_accs, valid_accs=val_accs)


        # Making predictions, obtaining new samples
        model.eval()
        labeled, unlabeled, best_predictions = evaluate_unlabeled(unlabeled[:1000], model, device)
        new_data, new_labels = labeled[:, :-1], labeled[:, -1]
        i += 1

        # Incorporating new samples into training array
        X_train, y_train = merge_into_training(X_train, y_train, new_data, new_labels)

