# This module wraps different baseline models
# LSTM, RNN, MLP and CONV1D
# includes a preprocessor layer

import torch
import torch.nn as nn
import torch.nn.functional as F
import Preprocessor as pp
import math


class LSTMLinear(nn.Module):

    def __init__(self, input_size, output_size, hidden_size,
                 n_layers, dropout=0.0):
        super(LSTMLinear, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.LSTM = nn.LSTM(1, hidden_size, n_layers, dropout=dropout)
        if isinstance(output_size, (list, tuple)):
            # YVG NOTE: i think it's for both multi-task learning and single task
            self.linear = nn.ModuleList(
                [nn.Linear(n_layers * hidden_size, o) for o in output_size]
            )
        else:
            self.linear = nn.Linear(n_layers * hidden_size, output_size)

    def forward(self, x):
        x = self.preprocess(x)
        # YVG NOTE: tensor.permute() is only used to swap the axes, not reshape
        output, hidden_T = self.LSTM(x.permute(2, 0, 1))

        # YVG NOTE: self tensor is contiguous in memory in C order.
        hidden = hidden_T[0].permute(1, 0, 2).contiguous()

        # YVG NOTE: Holds submodules in a list.
        # ModuleList can be indexed like a regular Python list,
        # but modules it contains are properly registered, and will be visible by all Module methods.
        if isinstance(self.linear, nn.ModuleList):
            pred = [
                l(
                    hidden.view(hidden.size(0), -1)
                ) for i, l in enumerate(self.linear)
            ]
        else:
            pred = self.linear(hidden.view(hidden.size(0), -1))
        return pred


class RNNLinear(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, n_layers,
                 dropout=0.0):
        super(RNNLinear, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.rnn = nn.RNN(1, hidden_size, n_layers, dropout=dropout)
        if isinstance(output_size, (list, tuple)):
            self.linear = nn.ModuleList(
                [nn.Linear(n_layers * hidden_size, o) for o in output_size]
            )
        else:
            self.linear = nn.Linear(n_layers * hidden_size, output_size)

    def forward(self, x):
        x = self.preprocess(x)
        output, hidden_T = self.rnn(x.permute(2, 0, 1))
        hidden_T = hidden_T.permute(1, 0, 2).contiguous()
        if isinstance(self.linear, nn.ModuleList):
            pred = [
                l(
                    hidden_T.view(hidden_T.size(0), -1)
                ) for i, l in enumerate(self.linear)
            ]
        else:
            pred = self.linear(hidden_T.view(hidden_T.size(0), -1))
        return pred


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.inlayer = nn.Linear(input_size, hidden_size)
        if isinstance(output_size, (list, tuple)):
            self.hiddenlayer = nn.ModuleList(
                [nn.Linear(hidden_size, o) for o in output_size]
            )
        else:
            self.hiddenlayer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.preprocess(x)
        x = torch.relu(self.inlayer(x))
        if isinstance(self.hiddenlayer, nn.ModuleList):
            pred = [
                l(x).squeeze(1) for i, l in enumerate(self.hiddenlayer)
            ]
        else:
            pred = self.hiddenlayer(x).squeeze(1)
        return pred


class Conv1DLinear(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 kernel_size=2,
                 pool_size=2
                 ):
        super(Conv1DLinear, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        # size of output
        lout = 3750 - kernel_size + 1
        self.pool1 = nn.MaxPool1d(pool_size)
        lout = math.floor(lout / pool_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = lout - kernel_size + 1
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = math.floor(lout / pool_size)
        if isinstance(output_size, (list, tuple)):
            self.out = nn.ModuleList(
                [nn.Linear(hidden_size * lout, o) for o in output_size]
            )
        else:
            self.out = nn.Linear(hidden_size * lout, output_size)
        self.nl = nn.ReLU()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.nl(self.pool1(self.conv1(x)))
        x = self.pool2(self.conv2(x))
        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(x.view(-1, x.size(1) * x.size(2))
                  ) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(x.view(-1, x.size(1) * x.size(2)))
        return pred


class Conv1DBNLinear(nn.Module):
    """YVG NOTE: based on the name, the network seems to be a combination of deep belief net and CNN"""
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 kernel_size=2,
                 pool_size=2,
                 dropout=0
                 ):
        super(Conv1DBNLinear, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.batch_norm0 = nn.BatchNorm1d(input_size)

        lout = 3750

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.pool1 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout5 = nn.Dropout(p=0.5)

        # YVG NOTE: prep for the next layer, reduce the hidden layer size by half
        # the last hidden is now the input layer to the next
        input_size = hidden_size
        hidden_size = hidden_size // 2

        self.conv3 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        # YVG NOTE: prep for the next layer
        # next hidden layer is the same size as the previous
        input_size = hidden_size

        self.conv5 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.pool3 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        # YVG NOTE: fully connected layer in the end
        if isinstance(output_size, (list, tuple)):
            self.out = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size * lout, 200),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(200, 200),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(200, o)
                    ) for o in output_size
                ]
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(hidden_size * lout, 200),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(200, output_size)
            )

        self.nl = nn.SELU()

    def l_out_conv1D(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation *
                 (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def l_out_maxpool1D(self, l_in, kernel_size, stride=None, padding=0,
                        dilation=1):
        if stride is None:
            stride = kernel_size
        l_out = self.l_out_conv1D(
            l_in, kernel_size, stride, padding, dilation
        )
        return l_out

    def forward(self, x):
        x = self.batch_norm0(self.preprocess(x))

        x = self.dropout(
            self.pool1(
                self.batch_norm1(self.nl(self.conv2(self.nl(self.conv1(x)))))
            )
        )

        x = self.dropout(
            self.pool2(
                self.batch_norm2(self.nl(self.conv4(self.nl(self.conv3(x)))))
            )
        )

        x = self.dropout(
            self.pool3(
                self.batch_norm3(self.nl(self.conv6(self.nl(self.conv5(x)))))
            )
        )

        if isinstance(self.out, nn.ModuleList):
            pred = [
                l(x.view(x.size(0), -1)) for i, l in enumerate(self.out)
            ]
        else:
            pred = self.out(x.view(x.size(0), -1))

        return pred
