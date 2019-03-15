import torch.nn as nn
from src.legacy.TABaseline.code import Preprocessor as pp


class Conv1DBNLinear(nn.Module):
    """Class 1 autoencoder, trains encoder and decoder together"""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        kernel_size=2,
        pool_size=2,
        dropout=0,
        label=True,
    ):
        super(Conv1DBNLinear, self).__init__()
        self.preprocess = pp.Preprocessor()

        self.batch_norm0 = nn.BatchNorm1d(input_size)
        self.label = label

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

        # prep for the next layer, reduce the hidden layer size by half
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

        # prep for the next layer
        # next hidden layer is the same size as the previous
        input_size = hidden_size

        self.conv5 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.pool3 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        # NOTE: fully connected layer in the end if labeled data, decoder if not
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
                        nn.Linear(200, o),
                    )
                    for o in output_size
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
                nn.Linear(200, output_size),
            )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * lout, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 3750),
        )

        self.nl = nn.SELU()

    def l_out_conv1D(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def l_out_maxpool1D(self, l_in, kernel_size, stride=None, padding=0, dilation=1):
        if stride is None:
            stride = kernel_size
        l_out = self.l_out_conv1D(l_in, kernel_size, stride, padding, dilation)
        return l_out

    def forward(self, x, label=True):
        x = self.preprocess(x)
        x = self.batch_norm0(x)

        x = self.dropout(
            self.pool1(self.batch_norm1(self.nl(self.conv2(self.nl(self.conv1(x))))))
        )

        x = self.dropout(
            self.pool2(self.batch_norm2(self.nl(self.conv4(self.nl(self.conv3(x))))))
        )

        x = self.dropout(
            self.pool3(self.batch_norm3(self.nl(self.conv6(self.nl(self.conv5(x))))))
        )

        if label:

            if isinstance(self.out, nn.ModuleList):
                pred = [l(x.view(x.size(0), -1)) for i, l in enumerate(self.out)]
            else:
                pred = self.out(x.view(x.size(0), -1))

        else:
            # Reconstruct input
            pred = self.decoder(x.view(x.size(0), -1))

        return pred
