from torch import nn, optim
from src.legacy.TABaseline.code import Preprocessor as pp


class AutoEncoder(nn.Module):
    """A regular fully connected Auto encoder"""

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.preprocess = pp.Preprocessor()

        # the number of hidden units are hardcoded for now.
        self.encoder = nn.Sequential(
            nn.Linear(3750, 2048), nn.ReLU(True), nn.Linear(2048, 1024), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(True), nn.Linear(2048, 3750)
        )

    def forward(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CnnAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=2, pool_size=2, dropout=0):
        super(CnnAutoEncoder, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.dropout = nn.Dropout(p=dropout)
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

        input_size = hidden_size
        hidden_size = hidden_size // 2

        self.conv3 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.pool2 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

        input_size = hidden_size

        self.conv5 = nn.Conv1d(input_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        lout = self.l_out_conv1D(lout, kernel_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.pool3 = nn.MaxPool1d(pool_size)
        lout = self.l_out_maxpool1D(lout, pool_size)

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

    def forward(self, x):
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

        pred = self.decoder(x.view(x.size(0), -1))

        return pred
