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


class Decoder(torch.nn.Module):
    def __init__(self, d_in, d_out, use_cuda, net_type, kernel_size):
        super(Decoder, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.use_cuda = use_cuda
        self.net_type = net_type
        self.kernel_size = kernel_size

        if self.use_cuda:
            # np.random.normal(0, 0.2) from bengio paper's initialization method
            # reference: Pezeshki,  Mohammad  et  al.  (2015).  “Deconstructing  the  Ladder  Network  Architecture”.
            # In:CoRRabs/1511.06430. arXiv:1511.06430.URL:http://arxiv.org/abs/1511.06430”
            self.a1 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a2 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a3 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a4 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a5 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())

            self.a6 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a7 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a8 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a9 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
            self.a10 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in).cuda())
        else:
            self.a1 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a2 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a3 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a4 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a5 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))

            self.a6 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a7 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a8 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a9 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))
            self.a10 = Parameter(np.random.normal(0, 0.2) * torch.ones(1, d_in))

        if self.d_out is not None:
            self.deconv = nn.Sequential(
                torch.nn.ConvTranspose1d(16, 32, self.kernel_size, stride=2),
                nn.LeakyReLU(negative_slope=0.05),
                torch.nn.ConvTranspose1d(16, 1, self.kernel_size, stride=2),
                nn.LeakyReLU(negative_slope=0.05),
                torch.nn.ConvTranspose1d(16, 1, self.kernel_size, stride=2),
            )
            if self.net_type == "cnn":
                self.V = self.deconv

            else:
                self.V = torch.nn.Linear(d_in, d_out, bias=False)
                self.V.weight.data = torch.randn(self.V.weight.data.size()) / np.sqrt(
                    d_in
                )

            # batch-normalization for u
            self.bn_normalize = torch.nn.BatchNorm1d(d_out, affine=False)

        # buffer for hat_z_l to be used for cost calculation
        else:
            self.buffer_hat_z_l = None

    def g(self, tilde_z_l, u_l):
        if self.use_cuda:
            ones = Parameter(torch.ones(tilde_z_l.size()[0], 1).cuda())
        else:
            ones = Parameter(torch.ones(tilde_z_l.size()[0], 1))

        # the following are trainable parameters used in the g( ) function defined in the paper
        # use as is
        b_a1 = ones.mm(self.a1)
        b_a2 = ones.mm(self.a2)
        b_a3 = ones.mm(self.a3)
        b_a4 = ones.mm(self.a4)
        b_a5 = ones.mm(self.a5)

        b_a6 = ones.mm(self.a6)
        b_a7 = ones.mm(self.a7)
        b_a8 = ones.mm(self.a8)
        b_a9 = ones.mm(self.a9)
        b_a10 = ones.mm(self.a10)

        u_l = torch.squeeze(u_l)

        # b_a1 =

        mu_l = (
            torch.mul(b_a1, torch.sigmoid(torch.mul(b_a2, u_l) + b_a3))
            + torch.mul(b_a4, u_l)
            + b_a5
        )

        v_l = (
            torch.mul(b_a6, torch.sigmoid(torch.mul(b_a7, u_l) + b_a8))
            + torch.mul(b_a9, u_l)
            + b_a10
        )

        if len(tilde_z_l.shape) == 3:
            tilde_z_l = torch.squeeze(tilde_z_l, dim=1)
        hat_z_l = torch.mul(tilde_z_l - mu_l, v_l) + mu_l

        hat_z_l = torch.unsqueeze(hat_z_l, dim=1)

        return hat_z_l

    def l_out_conv1D(self, l_in, kernel_size, stride=1, padding=0, dilation=1):
        l_out = (l_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride
        l_out = l_out + 1
        return int(l_out)

    def forward(self, tilde_z_l, u_l):

        hat_z_l = self.g(tilde_z_l, u_l)

        if self.d_out is not None:

            if len(hat_z_l.shape) == 2:
                hat_z_l = torch.unsqueeze(hat_z_l, dim=1)
            # do not change this to elif, not related to the one above

            t = self.V.forward(hat_z_l)

            t = torch.squeeze(t, dim=1)
            u_l_below = self.bn_normalize(t)

            u_l_below = torch.unsqueeze(u_l_below, dim=1)

            return u_l_below
        else:

            return None


class StackedDecoders(torch.nn.Module):
    def __init__(
        self, d_in, d_decoders, image_size, use_cuda, net_type_arr, kernel_size
    ):
        super(StackedDecoders, self).__init__()
        self.bn_u_top = torch.nn.BatchNorm1d(d_in, affine=False)
        self.decoders_ref = []
        self.decoders = torch.nn.Sequential()
        self.use_cuda = use_cuda
        n_decoders = len(d_decoders)
        self.net_type_arr = net_type_arr
        self.kernel_size = kernel_size
        for i in range(n_decoders):
            if i == 0:
                d_input = d_in

            else:
                d_input = d_decoders[i - 1]
            d_output = d_decoders[i]
            decoder_ref = "decoder_" + str(i)
            print(decoder_ref, d_input, d_output)
            decoder = Decoder(
                d_input, d_output, use_cuda, self.net_type_arr[i], self.kernel_size
            )
            self.decoders_ref.append(decoder_ref)
            self.decoders.add_module(decoder_ref, decoder)

        self.bottom_decoder = Decoder(image_size, None, use_cuda, "mlp", 0)

    def forward(self, tilde_z_layers, u_top, tilde_z_bottom):
        """
        Please refer to the paper for math details
        Calculates the reconstruction of input in decoders according to the algorithm proposed in the paper
        :param tilde_z_layers: z with noise
        :param u_top: a weight matrix used in the g () function
        :param tilde_z_bottom: output z with noise at the bottom decoder
        :return: reconstruction z_hat
        """
        # Note that tilde_z_layers should be in reversed order of encoders
        hat_z = []
        # print("u_top", u_top)
        u = self.bn_u_top(u_top)
        # u= u_top
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            tilde_z = tilde_z_layers[i]

            u = decoder.forward(tilde_z, u)
            hat_z.append(decoder.buffer_hat_z_l)
        self.bottom_decoder.forward(tilde_z_bottom, u)
        hat_z_bottom = self.bottom_decoder.buffer_hat_z_l.clone()

        hat_z.append(hat_z_bottom)

        return hat_z

    def bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        """
        batch norm for each z_hat layer
        :param hat_z_layers: reconstruction from decoders
        :param z_pre_layers: input to reconstruction
        :return: z_hat after batch norm
        """
        assert len(hat_z_layers) == len(z_pre_layers)
        hat_z_layers_normalized = []
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_layers, z_pre_layers)):
            if self.use_cuda:
                ones = Variable(torch.ones(z_pre.size()[0], 1).cuda())
            else:
                ones = Variable(torch.ones(z_pre.size()[0], 1))

            z_pre = torch.squeeze(z_pre)
            mean = torch.mean(z_pre, 0)
            noise_var = np.random.normal(loc=0.0, scale=1 - 1e-10, size=z_pre.size())
            if self.use_cuda:
                var = np.var(z_pre.data.cpu().numpy() + noise_var, axis=0).reshape(
                    1, z_pre.size()[1]
                )
            else:
                var = np.var(z_pre.data.numpy() + noise_var, axis=0).reshape(
                    1, z_pre.size()[1]
                )
            var = Variable(torch.FloatTensor(var))

            # convert back to cpu() to calculate and then convert back to cuda()
            # From code referenced online
            if self.use_cuda:
                hat_z = hat_z.cpu()
                ones = ones.cpu()
                mean = mean.cpu()

            mean = mean.view(1, mean.shape[0])

            hat_z_normalized = torch.div(
                hat_z - ones.mm(mean), ones.mm(torch.sqrt(var + 1e-10))
            )
            if self.use_cuda:
                hat_z_normalized = hat_z_normalized.cuda()
            hat_z_layers_normalized.append(hat_z_normalized)
        return hat_z_layers_normalized
