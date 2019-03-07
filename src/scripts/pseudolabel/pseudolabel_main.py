import numpy as np
import argparse
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import get_dataloader

from src.scripts.dataloader_utils import import_train_valid
from src.scripts.dataloader_utils import import_OM
from src.scripts.pytorch_utils import train_network

from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNClassification
from src.legacy.TeamB1pomt5.code.omsignal.utils.augmentation import RandomCircShift, RandomDropoutBurst, RandomNegate, RandomReplaceNoise
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import log_training, get_id_mapping, map_ids

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax


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
def merge_into_training(train_data, train_label, new_train_data, new_train_label, shuffle = True):
    """

    :param train_data: n samples x 1 channel x 3751 dimensions, previously used training dataset
    :param new_labeled_data: k samples x 1 channel x 3751 dimensions newly labeled training data
    :return:
    """
    if len(new_train_data) == 0:
        return train_data, train_label
    else:
        train_data = np.concatenate((train_data, new_train_data), axis=0)
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
    # unlabeled_data = unlabeled_data.astype(np.float32)

# Modify, supply the unlabeled_data by batch
    for idx, sample in enumerate(unlabeled_data):
        output = model(sample[0].cuda())
        sum__ = softmax(torch.exp(output[0]))
        sum_ = torch.sum(softmax(torch.exp(output[0])))
        pred_prob, pred_class = torch.max(torch.exp(output.data), 1)
        # best_preds = [pred_prob[i] for i in pred_prob if i > threshold]
        best_preds.extend([pred_prob[i].cpu().numpy() for i, ex in enumerate(pred_prob) if ex > threshold])
        # best_preds_label = [pred_class[i] for i in pred_prob if i > threshold]
        best_preds_label.extend([pred_class[i].cpu().numpy() for i, ex in enumerate(pred_prob) if ex > threshold])
        # best_samples = [unlabeled_data[i] for i in pred_prob if i > threshold]
        best_samples.extend([sample[0][i].cpu().numpy() for i, ex in enumerate(pred_prob) if ex > threshold])
        best_idx = [i for i, ex in enumerate(pred_prob) if i > threshold]
            # labeled.append(sample[0].data)
            # pred.append(pred_class[0])
            # best_preds.append((idx, pred_prob[0], pred_class[0]))
            # labeled_idx.append(idx)

        # unlabeled = [unlabeled_data[i] for i in range(1, len(unlabeled_data)) if i not in best_idx]
        unlabeled.extend([sample[0][i].cpu().numpy() for i, ex in enumerate(pred_prob) if ex <= threshold])
    if  len(best_preds) == 0:
        return [], np.array(unlabeled)
    else:
        best_samples, best_preds_label = np.array(best_samples), np.array(best_preds_label)
        best_samples, best_preds_label = (best_samples).reshape(np.shape(best_samples)[0], 1, np.shape(best_samples)[-1]),\
                                         (best_preds_label)[:, np.newaxis, np.newaxis]
        x1 = unlabeled
        return np.concatenate((best_samples, best_preds_label), axis=2), np.array(unlabeled)

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
    unlabeled = import_OM("unlabeled", cluster=cluster, len=5000)
    unlabeled = unlabeled[:,np.newaxis,:].astype(np.float32)
    unlabeled = preprocess(torch.from_numpy(unlabeled)).numpy()

    unlabeled_dataloader = DataLoader(TensorDataset(torch.Tensor(unlabeled)), batch_size=100, shuffle=False, num_workers=0)

    # Add transformations
    trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomNegate(0.5), \
        RandomReplaceNoise(0.5), RandomDropoutBurst(0.5)])


    epochs = 300
    loss_function = torch.nn.NLLLoss()

    i = 0
    while i < 3:

        # Creating dataloaders
        train_loader, valid_loader = fetch_dataloaders(X_train, y_train, X_valid, y_valid, transform=trsfrm, batch_size=32)

        # Creating model and dataloaders
        model = CNNClassification(3750, 32, conv1_num_filters=16, conv2_num_filters=32, conv_ksize=32, conv_stride=1,
                                  conv_padding=4,
                                  pool_ksize=5, pool_stride=8, pool_padding=1, num_linear=512, p=0.5)
        model.to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=1E-4, lr=0.01)

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
        labeled, unlabeled = evaluate_unlabeled(unlabeled_dataloader, model, device, threshold=0.8)
        if len(labeled) != 0:
            new_data, new_labels = labeled[:,:, :-1], labeled[:,:, -1]
            i += 1

            # Incorporating new samples into training array
            X_train, y_train = merge_into_training(X_train, y_train, new_data, new_labels)

