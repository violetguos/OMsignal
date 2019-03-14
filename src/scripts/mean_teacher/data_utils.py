import os
import itertools
import threading
import logging
import time
import torch
import math
import numpy as np

from datetime import datetime
from pandas import DataFrame
from collections import defaultdict

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, Sampler
from sklearn.preprocessing.data import normalize

from src.legacy.TABaseline.code.utils.memfile_utils import read_memfile
from src.legacy.TABaseline.code import data_augmentation as da



## Global Variable ##
NO_LABEL = 0
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CLUSTER_DIR = "/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/"

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
    hyperparam_data['fft'] = \
        bool(config.get('data', 'fft'))


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

def import_OM(dataset, cluster=True, len=50000, normalize_data=True): # Ideally replacing in legacy code : single function to import either unlabeled or labeled
    """
    Reads in the unlabeled data from the original memmap, separating by type.

    :param dataset: (string) dataset type : train, valid, unlabeled
    :param cluster: (bool) location of the data (in case of dummy)
    :param len: (int) amount of unlabeled example to import
    :return: desired dataset
    """
    labeled = False
    switch = {"Train": "TrainLabeled", "train": "TrainLabeled", "Training":"TrainLabeled",
              "valid": "ValidationLabeled", "Valid": "ValidationLabeled", "Validation": "ValidationLabeled"}
    dataset = dataset.capitalize()  # 'Train', 'Validation'
    if dataset in ["Train", "Validation", "Training", "Valid"]:
        dataset_ = switch[dataset]
        labeled = True
    elif dataset in ["Unlabeled"]:
        dataset_ = dataset

    if cluster:
        path_ = os.path.join(CLUSTER_DIR, 'MILA_{}Data.dat'.format(dataset_))
    else:
        path_ = os.path.join(DATA_DIR, 'MILA_{}Data_dummy.dat'.format(dataset_))

    if labeled:
        shape_ = (160, 3754)
    else:
        shape_ = (len, 3750)

    data = read_memfile(path_, shape=shape_, dtype='float32')

    # Split the data according to what's in the columns
    if labeled:
        ecg = data[:, :3750]
        if normalize_data:
            ecg = normalize(ecg)
        ecg = ecg.reshape(ecg.shape[0], 1, ecg.shape[1])
        PR_means = data[:, 3750].reshape((-1, 1))
        RT_means = data[:, 3751].reshape((-1, 1))
        RR_stdevs = data[:, 3752].reshape((-1, 1))
        ids = data[:, 3753].reshape((-1, 1))

        return ecg, PR_means, RT_means, RR_stdevs, ids
    else:
        if normalize_data:
            data = normalize(data)
        return data

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

        signal_len = np.shape(data)[2] - 4
        ecg = data[:,0,:signal_len]
        pr_mean = data[:, 0, signal_len]
        rt_mean = data[:, 0, signal_len+1]
        rr_stdev = data[:, 0, signal_len+2]
        userid = data[:, 0, signal_len+3]

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
        # if np.random.random() > 0.5:
        #     x = da.upside_down_inversion(x)
        # shift the series by 1 to 25 steps
        if np.random.random() > 0.5:
            x = da.shift_series(x, shift=np.random.randint(1, 26))
        # add partial gaussian noise 50% of the time
        if np.random.random() > 0.5:
            second_ = math.floor(len(x) / 125) - 2
            x = da.adding_partial_noise(
                x, second=np.random.randint(0, second_),
                duration=np.random.randint(1, 3)
            )

        return torch.Tensor(x).float()


def create_DataLoaders(train_data, valid_data, task='userid', batch_size = 256, labeled_batch_size = 32, include_unlabeled = False,
                       unlabeled_data = None, num_workers = 0, use_transform = False): # Need to lace training dataset with unlabeled examples
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

## Functions copied from original Mean Teacher Repository, https://github.com/CuriousAI/mean-teacher ##

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class TrainLog:
    """Saves training logs in Pandas msgpacks"""

    INCREMENTAL_UPDATE_TIME = 300

    def __init__(self, directory, name):
        self.log_file_path = "{}/{}.msgpack".format(directory, name)
        self._log = defaultdict(dict)
        self._log_lock = threading.RLock()
        self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record_single(self, step, column, value):
        self._record(step, {column: value})

    def record(self, step, col_val_dict):
        self._record(step, col_val_dict)

    def save(self):
        df = self._as_dataframe()
        df.to_msgpack(self.log_file_path, compress='zlib')

    def _record(self, step, col_val_dict):
        with self._log_lock:
            self._log[step].update(col_val_dict)
            if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
                self._last_update_time = time.time()
                self.save()

    def _as_dataframe(self):
        with self._log_lock:
            return DataFrame.from_dict(self._log, orient='index')


class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, run_idx):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        runner_name = os.path.basename(runner_file).split(".")[0]
        self.result_dir = "{root}/{runner_name}/{date:%Y-%m-%d_%H-%M-%S}/{run_idx}".format(
            root='results',
            runner_name=runner_name,
            date=datetime.now(),
            run_idx=run_idx
        )
        self.transient_dir = self.result_dir + "/transient"
        os.makedirs(self.result_dir)
        os.makedirs(self.transient_dir)

    def create_train_log(self, name):
        return TrainLog(self.result_dir, name)