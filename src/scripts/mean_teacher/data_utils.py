from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from src.legacy.meanteacher.pytorch.mean_teacher.data import TwoStreamBatchSampler

import torch
from src.legacy.TABaseline.code import data_augmentation as da
import numpy as np

## Global Variable ##
NO_LABEL = 0

def get_hyperparameters(config):
    """
    Function reading the different hyperparameters from the configuration file
    :param config: config object
    :return: different hyperparameter dictionnaries
    """
    hyperparam_data = {}
    hyperparam_optimizer = {}
    hyperparam_dataloader = {}
    hyperparam_model = {}
    hyperparam_meanteacher = {}
    hyperparam_path = {}

    hyperparam_data['normalize'] = \
        bool(config.get('data', 'normalize'))


    hyperparam_optimizer['learning_rate'] = \
        float(config.get('optimizer', 'learning_rate'))
    hyperparam_optimizer['initial_lr'] = \
        float(config.get('optimizer', 'initial_lr'))
    hyperparam_optimizer['lr_rampup'] = \
        int(config.get('optimizer', 'lr_rampup'))
    hyperparam_optimizer['lr_rampdown'] = \
        int(config.get('optimizer', 'lr_rampdown'))
    hyperparam_optimizer['epochs'] = \
        int(config.get('optimizer', 'nepoch'))
    hyperparam_optimizer['weight_decay'] = \
        float(config.get('optimizer', 'weight_decay'))


    hyperparam_dataloader['batchsize'] = \
        int(config.get('dataloader', 'batch_size'))
    hyperparam_dataloader['labeled_batch_size'] = \
        int(config.get('dataloader', 'labeled_batch_size'))
    hyperparam_dataloader['augment'] = \
        bool(config.get('dataloader', 'augment'))


    hyperparam_model['model'] = \
        config.get('model', 'name')
    hyperparam_model['hidden_size'] = \
        int(config.get('model', 'hidden_size'))
    hyperparam_model['dropout'] = \
        float(config.get('model', 'dropout'))
    hyperparam_model['n_layers'] = \
        int(config.get('model', 'n_layers'))
    hyperparam_model['kernel_size'] = \
        int(config.get('model', 'kernel_size'))
    hyperparam_model['pool_size'] = \
        int(config.get('model', 'pool_size'))
    hyperparam_model['dilation'] = \
        int(config.get('model', 'dilation'))


    hyperparam_meanteacher['EMA_decay'] = \
        float(config.get('meanteacher', 'EMA_decay'))
    hyperparam_meanteacher['consistency'] = \
        bool(config.get('meanteacher', 'consistency'))
    hyperparam_meanteacher['consistency_rampup'] = \
        int(config.get('meanteacher', 'consistency_rampup'))
    hyperparam_meanteacher['consistency_type'] = \
        (config.get('meanteacher', 'consistency_type'))
    hyperparam_meanteacher['checkpoint_epochs'] = \
        int(config.get('meanteacher', 'checkpoint_epochs'))
    hyperparam_meanteacher['evaluation_epochs'] = \
        int(config.get('meanteacher', 'evaluation_epochs'))


    hyperparam_path['tbpath'] = \
        config.get('path', 'tensorboard')
    hyperparam_path['modelpath'] = \
        config.get('path', 'model')

    return hyperparam_data, hyperparam_optimizer, hyperparam_dataloader, hyperparam_model, \
            hyperparam_meanteacher, hyperparam_path


def compress_id(raw_ids):
    # ids values run up to 43, but only 32 are present in the dataset
    # takes the raw ids as input, returns ids running from 0 to 31
    useridset = set(int(a) for a in raw_ids.flatten())
    userIdList = list(useridset)
    userIdList.sort()
    userid_dict = {a: i for i, a in enumerate(userIdList)}
    for i, raw_id in enumerate(raw_ids):
        raw_ids[i] = userid_dict[raw_id]
    return raw_ids


def norm_label(labels):
    labels = (labels - np.mean(labels)) / (np.std(labels) + 1e-5)
    return labels


def read_data(filename):
    # read binary data and return as a numpy array
    fp = np.memmap(filename, dtype='float32', mode='r', shape=(160, 3754))
    trainData = np.zeros(shape=(160, 3754))
    trainData[:] = fp[:]
    del fp
    raw_ids = trainData[:, 3753]
    pr_mean = trainData[:, 3750]
    rt_mean = trainData[:, 3751]
    rr_stdev = trainData[:, 3752]
    return trainData[:, :3750], pr_mean, rt_mean, rr_stdev, raw_ids


class ECGDataset(torch.utils.data.Dataset):

    def __init__(self, data, unlabeled=True, use_transform=False, target='userid'):
        """
        Parameters
        ----------
        filepath : string
           dataset location
        data_type : string, optional
           train, valid, or test
           if train, apply some transformation
        target : string
           pr_mean, rt_mean, rr_stdev, userid,
           or any comma separated combination of each of these strings.
        """
        self.use_transform = use_transform
        self.target_labels = target.split(",")
        self.target_labels = [s.lower().strip() for s in self.target_labels]

        possible_values = set(['pr_mean', 'rt_mean', 'rr_stdev', 'userid'])
        for s in self.target_labels:
            assert (s in possible_values)

        ecg = data[:,0,:3750]
        pr_mean = data[:, 0, 3750]
        rt_mean = data[:, 0, 3751]
        rr_stdev = data[:, 0, 3752]
        userid = data[:, 0, 3753]

        self.ecg = ecg
        self.targets = {'pr_mean': pr_mean,
                        'rt_mean': rt_mean,
                        'rr_stdev': rr_stdev,
                        'userid': userid
                        }

        useridset = set(int(a) for a in userid.flatten())
        self.userIdList = list(useridset)
        self.userIdList.sort()
        if unlabeled == False:
            self.userIdList = [-1] + self.userIdList
        self.userid_dict = {a: i for i, a in enumerate(self.userIdList)}

    def __len__(self):
        """Get the number of ecgs in the dataset.

        Returns
        -------
        int
           The number of ecg in the dataset.
        """
        return len(self.ecg)

    def __getitem__(self, index):

        """Get the items : ecg, target (userid by default)

        Parameters
        ----------
        index : int
           Index

        Returns
        -------
        img : tensor
           The ecg
        target : int or float, or a tuple
           When int, it is the class_index of the target class.
           When float, it is the value for regression
        """

        ecg = self.ecg[index]
        label = [
            torch.Tensor([self.userid_dict[int(self.targets[s][index])]]).type(torch.int64)
            if s == 'userid'
            else torch.Tensor([self.targets[s][index]]).float()
            for s in self.target_labels
        ]
        if len(label) == 1:
            label = label[0]

        if self.use_transform:
            ecg = self.train_transform(ecg)[np.newaxis,:], self.train_transform(ecg)[np.newaxis,:]
        else:
            ecg = torch.Tensor(ecg)[np.newaxis,:].float(), torch.Tensor(ecg)[np.newaxis,:].float()

        return ecg, label

    def train_transform(self, x):
        """
        Take an ECG numpy array x
        Randomly flip it, shift it and noise it
        """
        # first, random flip
        if np.random.random() > 0.5:
            x = da.upside_down_inversion(x)
        # shift the series by 1 to 25 steps
        if np.random.random() > 0.5:
            x = da.shift_series(x, shift=np.random.randint(1, 26))
        # add partial gaussian noise 50% of the time
        if np.random.random() > 0.5:
            x = da.adding_partial_noise(
                x, second=np.random.randint(0, 29),
                duration=np.random.randint(1, 3)
            )
        # # add gaussian noise 50% of the time
        # if np.random.random()>0.5:
        #     x = da.adding_noise(x,epsilon=0.01)
        return torch.Tensor(x).float()


def create_DataLoaders(train_data, valid_data, task='userid', batch_size = 256, labeled_batch_size = 32, include_unlabeled = False,
                       unlabeled_data = None, num_workers = 0, use_transform = True): # Need to lace training dataset with unlabeled examples
    """
    Function allowing construction of labeled and unlabeled training dataloaders, with TwoStreamBatchSampler function.
    No unlabeled data is found in the validation loader.
    :param train_data: (np.array) training data
    :param valid_data: (np.array) validation data
    :param target: (string) labels to keep. Not recommended to include more than one label type
    :param batch_size: (int) total batch size, for both training and valid
    :param labeled_batch_size: (int) amount of labeled data within training batch dize
    :param include_unlabeled: (bool) include unlabeled data or not. If not, training loader only has labeled data
    :param unlabeled_data: (np.array) unlabeled data
    :param num_workers: (int) depends on hardware ressources
    :param use_transform: (bool) Add data augmentation to fetching process
    :return: Train and Valid dataloaders
    """
    unlabeled_train_data = train_data
    if include_unlabeled:
        unlabeled_train_data = np.concatenate((train_data, unlabeled_data), axis=0)
        np.random.shuffle(unlabeled_train_data)

    unlabeled_idx = [i for i,ex in enumerate(unlabeled_train_data) if ex[:,-1]== -1]
    labeled_idx = [i for i, ex in enumerate(unlabeled_train_data) if ex[:, -1] != -1]

    if include_unlabeled:
        batch_sampler = TwoStreamBatchSampler(unlabeled_idx, labeled_idx, batch_size, labeled_batch_size)
    else:
        sampler = SubsetRandomSampler(labeled_idx)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)

    train_dataset = ECGDataset(unlabeled_train_data, target=task, use_transform=use_transform)
    valid_dataset = ECGDataset(valid_data, unlabeled=False, target=task, use_transform=use_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=num_workers,
                                               pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, valid_loader
