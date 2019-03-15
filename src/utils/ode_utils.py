import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import log_training, get_id_mapping, map_ids
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import import_train_valid, import_OM,OM_dataset,get_dataloader
from src.legacy.TeamB1pomt5.code.omsignal.utils.augmentation import RandomCircShift, RandomDropoutBurst, RandomNegate, RandomReplaceNoise



def get_hyperparameters(config):
    """
    :param config: an .in file with params for a model
    :param autoencoder: Boolean, to include an autoencoder in CNN or not
    :return: a hyperparam dictionary
    """
    hyperparam = {}
    hyperparam["learning_rate"] = float(config.get("optimizer", "learning_rate"))
    hyperparam["batchsize"] = int(config.get("optimizer", "batch_size"))
    hyperparam["nepoch"] = int(config.get("optimizer", "nepoch")) 
    hyperparam["model"] = config.get("model", "name")
    hyperparam["subsample"] = config.get("model", "include_subsamples")
    hyperparam["tbpath"] = config.get("path", "tensorboard")
    hyperparam["modelpath"] = config.get("path", "model")
    return hyperparam



def load_data(model_hp_dict,gpu_avail,device):
    # get data 
    X_train, X_valid, y_train, y_valid = import_train_valid('ids', cluster=gpu_avail)

    # Remapping IDs

    

    #process & format

    y_train,y_valid = get_map_ids(y_train,y_valid)

    preprocess = Preprocessor()
    preprocess.to(device)

    X_train = preprocess(torch.from_numpy(X_train)).numpy()
    X_valid = preprocess(torch.from_numpy(X_valid)).numpy()
    
    # Add transformations
    trsfrm = transforms.RandomChoice([RandomCircShift(0.5), RandomNegate(0.5), \
        RandomReplaceNoise(0.5), RandomDropoutBurst(0.5)])

    train_loader, valid_loader = fetch_dataloaders(X_train, y_train, X_valid, y_valid,model_hp_dict['batchsize'], transform=trsfrm)


    # unlabeled = import_OM("unlabeled")
    #unlabeled = unlabeled[:,np.newaxis,:]
    # unlabeled = preprocess(torch.from_numpy(unlabeled)).numpy()    
    # unlabeled_loader = get_dataloader_unlabeled(unlabeled,trsfrm, model_hp_dict['batchsize'], shuffle=False)

    return train_loader, valid_loader#, unlabeled_loader







def fetch_dataloaders(train_data, train_labels, valid_data, valid_labels, batch_size=50, transform=None): #No need with train_ID_CNN
    """
    fetch_dataloders is a function which creates the dataloaders required for data training
    :param train_data: n samples x 1 channel x m dimensions (3750 + 4/1)
    :param valid_data:
    :param unlabeled_data:
    :return: dataloaders of input data
    """
    train_loader = get_dataloader(train_data, train_labels, transform, batch_size=batch_size)
    valid_loader = get_dataloader(valid_data, valid_labels, transform, batch_size=batch_size) #Only uses original validation data

    return train_loader, valid_loader

def training_loop(training_dataloader, validation_dataloader, model):
    """

    :param training_dataloader:
    :param validation_dataloader:
    :param model:
    :return: model
    """
    return model

def get_map_ids(y_train,y_valid):
    mapping = get_id_mapping(y_train[:,0])
    y_train[:,0] = map_ids(y_train[:,0], mapping)
    y_valid[:,0] = map_ids(y_valid[:,0], mapping)
    return y_train,y_valid


def get_dataloader_unlabeled(X,transform, batch_size, shuffle=True, task_type = "Regression"):
    
    if task_type not in ["Regression", "Classification" , "Ranking"]:
        raise ValueError("task_type must be in ['Regression', 'Classification' , 'Ranking']")

    if task_type == "Ranking":
        dataset = Rank_dataset(X, y, transform=transform)
    elif task_type == "Classification":
        # X is n,1,3750, we want to calculate the FFT over the last axis
        arr_copy = np.copy(X)
        arr_copy = np.fft.rfft(arr_copy, axis=2).astype(np.float32)
        dataset = OM_dataset(arr_copy, y, transform=transform)    
    else:
        placeholder = np.zeros([np.shape(X[:,:,])[0],1])
        y = placeholder;
        dataset = OM_dataset(X, y, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader