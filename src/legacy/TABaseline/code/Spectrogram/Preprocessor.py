import torch
import torch.nn as nn
import torch.nn.functional as Functional


class Preprocessor(nn.Module):

    def __init__(self):

        super(Preprocessor, self).__init__()

        # Kernel size to use for moving average baseline wander removal: 2
        # seconds * 125 HZ sampling rate, + 1 to make it odd

        self.maKernelSize = (2 * 125) + 1

        # Kernel size to use for moving average baseline wander removal: 4
        # seconds * 125 HZ sampling rate , + 1 to make it odd

        self.mvKernelSize = (4 * 125) + 1

    def forward(self, x):

        with torch.no_grad():

            # Remove window mean and standard deviation

            x = (x - torch.mean(x, dim=2, keepdim=True)) / \
                (torch.std(x, dim=2, keepdim=True) + 0.00001)

            # Moving average baseline wander removal

            x = x - Functional.avg_pool1d(x, kernel_size=self.maKernelSize,
                                          stride=1, padding=(self.maKernelSize - 1) // 2)

            # Moving RMS normalization

            x = x / (torch.sqrt(Functional.avg_pool1d(torch.pow(x, 2), kernel_size=self.mvKernelSize,
                                                      stride=1, padding=(self.mvKernelSize - 1) // 2)) + 0.00001)

        # Don't backpropagate further

        x = x.detach().contiguous()

        return x
