import torch
from torch.utils.data import sampler, DataLoader
from src.legacy.TABaseline.code import data_augmentation as da
import numpy as np


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

    def __init__(self, filepath, use_transform=False, target='userid'):
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

        # No need of double targets
        assert(len(set(self.target_labels)) == len(self.target_labels))

        ecg, pr_mean, rt_mean, rr_stdev, userid = read_data(filepath)

        useridset = set(int(a) for a in userid.flatten())
        self.userIdList = list(useridset)
        self.userIdList.sort()
        self.userid_dict = {a: i for i, a in enumerate(self.userIdList)}

        self.ecg = ecg
        self.targets = {'pr_mean': pr_mean,
                        'rt_mean': rt_mean,
                        'rr_stdev': rr_stdev,
                        'userid': userid
                        }

    def __len__(self):
        """Get the number of ecgs in the dataset.

        Returns
        -------
        int
           The number of ecg in the dataset.
        """
        return len(self.ecg)

    def __getitem__(self, index):
        """YVG NOTES: return the idx here for samplers to record index"""

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
            torch.Tensor(
                [self.userid_dict[int(self.targets[s][index])]]
            ).type(torch.int64) if s == 'userid'
            else torch.Tensor([self.targets[s][index]]).float()
            for s in self.target_labels
        ]
        if len(label) == 1:
            label = label[0]

        if self.use_transform:
            ecg = self.train_transform(ecg)
        else:
            ecg = torch.Tensor(ecg).float()

        return ecg.unsqueeze(0), label

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
