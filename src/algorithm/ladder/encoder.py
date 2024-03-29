import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn

"""
Note: this is adapted for Ladder with convolution from open
source implementation of ladder with MLP from
https://github.com/abhiskk/ladder
"""


class Encoder(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        activation_type,
        train_bn_scaling,
        noise_level,
        use_cuda,
        net_type,
        kernel_size,
    ):
        super(Encoder, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation_type = activation_type
        self.train_bn_scaling = train_bn_scaling
        self.noise_level = noise_level
        self.use_cuda = use_cuda
        self.kernel_size = kernel_size
        # added to commendate different types of layer architecture
        self.net_type = net_type

        # Encoder
        # Encoder only uses W matrix, no bias

        self.linear = torch.nn.Linear(d_in, d_out, bias=False)

        self.conv = nn.Sequential(
            torch.nn.Conv1d(1, 16, self.kernel_size, stride=2),
            torch.nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.05),
            torch.nn.Conv1d(16, 32, self.kernel_size, stride=2),
            torch.nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.05),
        )

        self.pool = torch.nn.MaxPool1d(3, return_indices=True)
        self.linear.weight.data = torch.randn(self.linear.weight.data.size()) / np.sqrt(
            d_in
        )
        # Batch Normalization
        # For Relu Beta of batch-norm is redundant, hence only Gamma is trained
        # For Softmax Beta, Gamma are trained
        # batch-normalization bias

        self.bn_normalize_clean = torch.nn.BatchNorm1d(d_out, affine=False)
        self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)
        if self.use_cuda:
            self.bn_beta = Parameter(torch.cuda.FloatTensor(1, d_out))
        else:
            self.bn_beta = Parameter(torch.FloatTensor(1, d_out))
        self.bn_beta.data.zero_()
        if self.train_bn_scaling:
            # batch-normalization scaling
            if self.use_cuda:
                self.bn_gamma = Parameter(torch.cuda.FloatTensor(1, d_out))
                self.bn_gamma.data = torch.ones(self.bn_gamma.size()).cuda()
            else:
                self.bn_gamma = Parameter(torch.FloatTensor(1, d_out))
                self.bn_gamma.data = torch.ones(self.bn_gamma.size())

        # Activation
        if activation_type == "relu":
            self.activation = torch.nn.LeakyReLU(negative_slope=0.05)
        elif activation_type == "softmax":
            # choose dim = 1 becuase the z before actication is shape (batch_size, 32)
            # where 32 is number of possible user IDs
            # we want dim=1 to sum to 1
            self.activation = torch.nn.Softmax(dim=1)

        else:
            raise ValueError("invalid Acitvation type")

        # buffer for z_pre, z which will be used in decoder cost
        self.buffer_z_pre = None
        self.buffer_z = None
        # buffer for tilde_z which will be used by decoder for reconstruction
        self.buffer_tilde_z = None

    def bn_gamma_beta(self, x):
        if self.use_cuda:
            ones = Parameter(torch.ones(x.size()[0], 1).cuda())
        else:
            ones = Parameter(torch.ones(x.size()[0], 1))
        t = x + ones.mm(self.bn_beta)
        if self.train_bn_scaling:
            t = torch.mul(t, ones.mm(self.bn_gamma))
        return t

    def forward_clean(self, h):
        if self.net_type == "mlp":
            z_pre = self.linear(h)
        elif self.net_type == "cnn":
            if len(h.shape) == 2:
                h = torch.unsqueeze(h, dim=1)
            z_pre = self.conv(h)

        # Store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_z_pre = z_pre.detach().clone()
        z_pre = z_pre.squeeze(1)
        z = self.bn_normalize_clean(z_pre)
        self.buffer_z = z.detach().clone()
        z_gb = self.bn_gamma_beta(z)

        h = self.activation(z_gb)

        return h

    def l_out_conv1D(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def forward_noise(self, tilde_h):
        """
        refer to the paper for the detailed algorithm
        :param tilde_h: data input or output from last encoder
        :return:
        """
        # z_pre will be used in the decoder cost
        if self.net_type == "mlp":
            z_pre = self.linear(tilde_h)
        elif self.net_type == "cnn":
            if len(tilde_h.shape) == 2:
                tilde_h = torch.unsqueeze(tilde_h, dim=1)

            z_pre = self.conv(tilde_h)

        # accomendate the 3 dim <-> 2 dim transformation for Convolutional
        z_pre = torch.squeeze(z_pre, dim=1)
        print("z_pre", z_pre.shape)
        z_pre_norm = self.bn_normalize(z_pre)
        # Add noise
        noise = np.random.normal(
            loc=0.0, scale=self.noise_level, size=z_pre_norm.size()
        )
        if self.use_cuda:
            noise = Variable(torch.cuda.FloatTensor(noise))
        else:
            noise = Variable(torch.FloatTensor(noise))
        # tilde_z will be used by decoder for reconstruction
        tilde_z = z_pre_norm + noise
        # store tilde_z in buffer
        self.buffer_tilde_z = tilde_z
        z = self.bn_gamma_beta(tilde_z)

        if self.activation_type != None:
            h = self.activation(z)
        else:
            h = z

        return h


class StackedEncoders(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_encoders,
        activation_types,
        train_batch_norms,
        noise_std,
        use_cuda,
        net_type_arr,
        kernel_size,
    ):
        super(StackedEncoders, self).__init__()
        self.buffer_tilde_z_bottom = None
        self.encoders_ref = []
        self.encoders = torch.nn.Sequential()
        self.noise_level = noise_std
        self.use_cuda = use_cuda
        n_encoders = len(d_encoders)
        self.kernel_size = kernel_size
        self.net_type_arr = net_type_arr

        # used to normalize the batch data
        self.data_batch_norm = torch.nn.BatchNorm1d(1)

        for i in range(n_encoders):
            if i == 0:
                d_input = d_in
            else:
                d_input = d_encoders[i - 1]

            d_output = d_encoders[i]
            activation = activation_types[i]
            train_batch_norm = train_batch_norms[i]
            encoder_ref = "encoder_" + str(i)
            print("create encoder number {}".format(i))
            encoder = Encoder(
                d_input,
                d_output,
                activation,
                train_batch_norm,
                noise_std,
                use_cuda,
                self.net_type_arr[i],
                kernel_size,
            )
            self.encoders_ref.append(encoder_ref)
            self.encoders.add_module(encoder_ref, encoder)

    def forward_clean(self, x):
        """
        forward pass with no noise injection
        :param x: data
        :return: compressed result from encoder
        """
        h = x
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_clean(h)
        return h

    def forward_noise(self, x):
        """
        forward pass with noise injection
        :param x: data
        :return: compressed result from encoder
        """
        noise = np.random.normal(loc=0.0, scale=self.noise_level, size=x.size())
        if self.use_cuda:
            noise = Variable(torch.cuda.FloatTensor(noise))
        else:
            noise = Variable(torch.FloatTensor(noise))

        if self.use_cuda:
            x = Variable(x.cuda())
        else:
            x = Variable(torch.FloatTensor(x))
        h = x + noise

        self.buffer_tilde_z_bottom = h.clone()

        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_noise(h)

        return h

    def get_encoders_tilde_z(self, reverse=True):
        """
        reads the tilde z parameter from encoders to do more math
        :param reverse: encoder is the reverse of decoder, use this to iterate through layers
        :return: list of tilde_z
        """
        tilde_z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            tilde_z = encoder.buffer_tilde_z.clone()
            tilde_z_layers.append(tilde_z)
        if reverse:
            tilde_z_layers.reverse()
        return tilde_z_layers

    def get_encoders_z_pre(self, reverse=True):
        """
        reads the z parameter from encoders to do more math
        :param reverse: encoder is the reverse of decoder, use this to iterate through layers
        :return: list of z_pre
        """
        z_pre_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z_pre = encoder.buffer_z_pre.clone()
            z_pre_layers.append(z_pre)
        if reverse:
            z_pre_layers.reverse()
        return z_pre_layers

    def get_encoders_z(self, reverse=True):
        """
        reads the z parameter from encoders to do more math
        :param reverse: encoder is the reverse of decoder, use this to iterate through layers
        :return: list of z
        """
        z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z = encoder.buffer_z.clone()
            z_layers.append(z)
        if reverse:
            z_layers.reverse()
        return z_layers
