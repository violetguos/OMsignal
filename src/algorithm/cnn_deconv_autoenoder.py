from src.legacy.TABaseline.code import Preprocessor as pp
from torch import nn
from src.utils import constants


class CnnDeconvAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size=2,
        pool_size=2,
        dropout=0,
    ):
        super(CnnDeconvAutoEncoder, self).__init__()
        self.preprocess = pp.Preprocessor()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm0 = nn.BatchNorm1d(input_size)
        l_out = constants.FFT_SHAPE[1]

        # encoders operations
        input_size_buf_block1 = input_size
        hidden_size_buf_block1 = hidden_size
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size)
        l_out = self.l_out_conv1d(l_out, kernel_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        l_out = self.l_out_conv1d(l_out, kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.pool1 = nn.MaxPool1d(pool_size, return_indices=True)
        l_out = self.l_out_maxpool1d(l_out, pool_size)

        # self.dropout = nn.Dropout(p=dropout)
        # self.dropout5 = nn.Dropout(p=0.5)

        input_size = hidden_size
        input_size_buf_block2 = input_size
        hidden_size = hidden_size // 2
        hidden_size_buf_block2 = hidden_size

        self.conv3 = nn.Conv1d(input_size, hidden_size, kernel_size)
        l_out = self.l_out_conv1d(l_out, kernel_size)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        l_out = self.l_out_conv1d(l_out, kernel_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.pool2 = nn.MaxPool1d(pool_size, return_indices=True)
        l_out = self.l_out_maxpool1d(l_out, pool_size)
        input_size = hidden_size
        input_size_buf_block3 = input_size
        hidden_size_buf_block3 = hidden_size

        self.conv5 = nn.Conv1d(input_size, hidden_size, kernel_size)
        l_out = self.l_out_conv1d(l_out, kernel_size)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        l_out = self.l_out_conv1d(l_out, kernel_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.pool3 = nn.MaxPool1d(pool_size, return_indices=True)
        l_out = self.l_out_maxpool1d(l_out, pool_size)
        final_encoder_l_out = l_out
        print("lout", l_out)


        self.prediction_layer = nn.Linear(
                final_encoder_l_out * hidden_size_buf_block3 ,
                constants.NUM_IDS
        )

        # decoder operations
        # input_size = hidden_size still holds here
        self.unpool3 = nn.MaxUnpool1d(pool_size)
        l_out = self.inv_l_out_maxpool1d(l_out, pool_size)
        self.deconv6 = nn.ConvTranspose1d(
            hidden_size_buf_block3, hidden_size_buf_block3, kernel_size
        )
        l_out = self.inv_l_out_conv1d(l_out, kernel_size)
        self.deconv5 = nn.ConvTranspose1d(
            hidden_size_buf_block3, input_size_buf_block3, kernel_size
        )
        l_out = self.inv_l_out_conv1d(l_out, kernel_size)

        #         input_size = hidden_size
        #         hidden_size = hidden_size // 2

        self.unpool2 = nn.MaxUnpool1d(pool_size)
        l_out = self.inv_l_out_maxpool1d(l_out, pool_size)
        self.deconv4 = nn.ConvTranspose1d(
            hidden_size_buf_block2, hidden_size_buf_block2, kernel_size
        )
        l_out = self.inv_l_out_conv1d(l_out, kernel_size)
        self.deconv3 = nn.ConvTranspose1d(
            hidden_size_buf_block2, input_size_buf_block2, kernel_size
        )
        l_out = self.inv_l_out_conv1d(l_out, kernel_size)

        self.unpool1 = nn.MaxUnpool1d(pool_size)
        l_out = self.inv_l_out_conv1d(l_out, kernel_size)
        self.deconv2 = nn.ConvTranspose1d(
            hidden_size_buf_block1, hidden_size_buf_block1, kernel_size
        )
        l_out = self.inv_l_out_conv1d(l_out, kernel_size)
        self.deconv1 = nn.ConvTranspose1d(
            hidden_size_buf_block1, input_size_buf_block1, kernel_size
        )



        # TODO: what is selu???
        self.nl = nn.SELU()

    def l_out_conv1d(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def inv_l_out_conv1d(
        self, l_in, kernel_size, stride=1, padding=0, dilation=1, output_padding=0
    ):
        """
        inverse of a conv1d
        Input: (N, C_{in}, L_{in})
        Output: (N, C_{out}, L_{out}) where
        Lout=(Lin−1)×stride−2×padding + dilation×(kernel_size−1) + output_padding + 1
        :return:
        """

        l_out = (
            (l_in - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding
            + 1
        )
        return l_out

    def l_out_maxpool1d(self, l_in, kernel_size, stride=None, padding=0, dilation=1):
        if stride is None:
            stride = kernel_size
        l_out = self.l_out_conv1d(l_in, kernel_size, stride, padding, dilation)
        return l_out

    def inv_l_out_maxpool1d(self, l_in, kernel_size, stride=None, padding=0):
        """
        calculates the size of inverse operation, MaxPool1d
        :param l_in: input size
        :param kernel_size: Size of the max pooling window.
        :param stride:  Stride of the max pooling window, defaults to kernel_size
        :param padding: default is 0 for normal pooling operation
        :return: inverse size of 1D unpool operation
        """
        if stride is None:
            stride = kernel_size

        l_out = (l_in - 1) * stride - 2 * padding + kernel_size
        return l_out

    def encoder(self, x):
        # encoder
        # print("x = self.conv1(x)",  x.size())

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)

        # Need the size before pool() as an argument to unpool() for stability
        out1 = list(x.size())
        x, idx1 = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch_norm2(x)

        out2 = list(x.size())
        x, idx2 = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.batch_norm3(x)

        out3 = list(x.size())
        x, idx3 = self.pool3(x)

        return x, idx1, idx2, idx3, out1, out2, out3

    def decoder(self, x, idx1, idx2, idx3, out1, out2, out3):
        # decoder
        x = self.unpool3(x, idx3, output_size=out3)
        x = self.deconv6(x)
        x = self.deconv5(x)
        x = self.unpool2(x, idx2, output_size=out2)
        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.unpool1(x, idx1, output_size=out1)
        x = self.deconv2(x)
        x = self.deconv1(x)
        # many online literatures have a tanh there so let's try it
        x = nn.Tanh()(x)
        return x

    def preprocess_norm(self, x, batch_size):
        x = x.view(len(x), constants.FFT_SHAPE[0], constants.FFT_SHAPE[1])

        x = self.preprocess(x)
        x = x.view(len(x), 1, constants.FFT_SHAPE[1])
        # print("x = self.batch_norm0(x)",  x.size())

        x = self.batch_norm0(x)


        return x

    def forward(self, x, prediction=False):

        if prediction:
            x, idx1, idx2, idx3, out1, out2, out3 = self.encoder(x)
            y = self.prediction_layer(x.view(x.size(0), -1))
            return x, y


        else:
            x, idx1, idx2, idx3, out1, out2, out3 = self.encoder(x)
            x = self.decoder(x,  idx1, idx2, idx3, out1, out2, out3)
            return x

