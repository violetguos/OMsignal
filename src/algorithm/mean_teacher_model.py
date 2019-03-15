import torch.nn as nn
from src.legacy.TABaseline.code import Preprocessor as pp
import math


class Conv1DLinear(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        kernel_size=2,
        pool_size=2,
        dropout=0.5,
    ):
        super(Conv1DLinear, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        # lout = size of output
        lout = 1876 - kernel_size + 1  # Change for 3750!
        self.pool1 = nn.MaxPool1d(pool_size)
        lout = math.floor(lout / pool_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size)
        lout = lout - (kernel_size) + 1
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = math.floor(lout / pool_size)
        if isinstance(output_size, (list, tuple)):
            self.out = nn.ModuleList(
                [nn.Linear(hidden_size * 2 * lout, o) for o in output_size]
            )
        else:
            self.out = nn.Linear(hidden_size * 2 * lout, output_size)
        self.nl = nn.ReLU()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.dropout(x)
        x = self.nl(self.pool1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool2(self.conv2(x))
        x = self.dropout(x)
        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(x.view(-1, x.size(1) * x.size(2))) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(x.view(-1, x.size(1) * x.size(2)))
        return pred
