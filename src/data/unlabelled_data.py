import torch
from torch.utils.data import sampler, DataLoader
from src.legacy.TABaseline.code import data_augmentation as da
import numpy as np
from src.utils import constants
from src.utils.os_helper import get_num_data_points


class UnlabelledDataset(torch.utils.data.Dataset):
    """A new dataset class for unlabelled datam since there's no label to return"""

    def __init__(self, data_path, use_transform=False):
        """
        Parameters
        ----------
        data_path : string
           dataset location
        """
        self.data_path = data_path
        self.number_of_data_points = get_num_data_points(
            data_path, constants.SIZE_OF_DATA_POINT_BYTES
        )

        self.use_transform = use_transform

    def __len__(self):
        """Get the number of ecgs in the dataset.

        Returns
        -------
        int
           The number of ecg in the dataset.
        """
        return self.number_of_data_points

    def __getitem__(self, idx):
        """NOTES: return the idx here for samplers to record index"""

        """Get the items : ecg, target (userid by default)

        Parameters
        ----------
        index : int
           Index

        Returns
        -------
        img : tensor
           The ecg
        idx: the input index, useful for subset sampler in the future
        """

        ecg = np.memmap(
            self.data_path,
            dtype="float32",
            mode="r",
            shape=constants.SHAPE_OF_ONE_DATA_POINT,
            offset=constants.SIZE_OF_DATA_POINT_BYTES * idx,
        )

        if self.use_transform:
            # ecg = self.train_transform(ecg)
            # print("in unlabelled", ecg.shape)
            arr_copy = np.copy(ecg)
            arr_copy = np.fft.rfft(arr_copy, axis=1).astype(np.float32)
            ecg = torch.Tensor(arr_copy).float()
            # print("ecg", ecg.size())

        else:
            ecg = torch.Tensor(ecg).float()

        return ecg.unsqueeze(0), idx

    def train_transform(self, x):
        """
        Take an ECG numpy array x
        Change datatype
        """
        return torch.Tensor(x).float()
