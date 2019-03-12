from __future__ import print_function

import numpy as np
import argparse
import os
import torch
import copy
from torch.autograd import Variable
from src.algorithm.encoder import StackedEncoders, l_out_conv1d
from src.algorithm.decoder import StackedDecoders, inv_l_out
import src.utils.constants as constants

from src.scripts.unsupervised_pretraining import load_data
from src.legacy.TABaseline.code import Preprocessor as pp
import matplotlib.pyplot as plt




def layer_plot(x, title="ladder", fig="ladder"):
    """
    :param x: a dictionary of (key, numpy arrays) to plot
    :param fig: name of the figure
    :return:
    """
    plt.title(title)

    for key, val in x.items():
        plt.plot(val.data.cpu().numpy().reshape(constants.SHAPE_OF_ONE_DATA_POINT[1]), label=key)


    plt.legend(loc='best')

    plt.savefig(fig + ".png")
    plt.show()
    plt.close()
    plt.clf()


class Ladder(torch.nn.Module):
    def __init__(
        self,
        encoder_sizes,
        decoder_sizes,
        encoder_activations,
        encoder_train_bn_scaling,
        noise_std,
        use_cuda,
        encoder_layer_type_arr,
        decoder_layer_type_arr,
        kernel_size
    ):
        super(Ladder, self).__init__()
        self.use_cuda = use_cuda
        decoder_in = encoder_sizes[-1]
        encoder_in = decoder_sizes[-1]
        self.se = StackedEncoders(
            encoder_in,
            encoder_sizes,
            encoder_activations,
            encoder_train_bn_scaling,
            noise_std,
            use_cuda,
            encoder_layer_type_arr,
            kernel_size
        )
        self.de = StackedDecoders(
            decoder_in, decoder_sizes, encoder_in, use_cuda, decoder_layer_type_arr,
            kernel_size
        )
        self.bn_image = torch.nn.BatchNorm1d(encoder_in, affine=False)
        self.kernel_size = kernel_size

    def forward_encoders_clean(self, data):
        layer_out = self.se.forward_clean(data)


        return layer_out

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        layer_out = self.de.forward(
            tilde_z_layers, encoder_output, tilde_z_bottom
        )
        return layer_out


    def get_encoders_tilde_z(self, reverse=True):
        return self.se.get_encoders_tilde_z(reverse)

    def get_encoders_z_pre(self, reverse=True):
        return self.se.get_encoders_z_pre(reverse)

    def get_encoder_tilde_z_bottom(self):
        return self.se.buffer_tilde_z_bottom.clone()

    def get_encoders_z(self, reverse=True):
        return self.se.get_encoders_z(reverse)

    def decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.de.bn_hat_z_layers(hat_z_layers, z_pre_layers)


def evaluate_performance(
    ladder,
    valid_loader,
    e,
    agg_cost_scaled,
    agg_supervised_cost_scaled,
    agg_unsupervised_cost_scaled,
    args,
):
    correct = 0.0
    total = 0.0
    for batch_idx, (data, target) in enumerate(valid_loader):
        if args.cuda:
            data = data.cuda()

        # a hack for now
        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        target = target[3]
        target = target.to(device=device)
        target = target.squeeze()

        output = ladder.forward_encoders_clean(data)
        # print("output" , output)
        # print("target", target)
        if args.cuda:
            output = output.cpu()
            target = target.cpu()

        output = output.detach().numpy()
        target = target.data.numpy()

        preds = np.argmax(output, axis=1)
        # print("preds", preds)
        correct += np.sum(target == preds)
        total += target.shape[0]


    print(
        "Epoch:",
        e + 1,
        "\t",
        "Total Cost:",
        "{:.4f}".format(agg_cost_scaled),
        "\t",
        "Supervised Cost:",
        "{:.4f}".format(agg_supervised_cost_scaled),
        "\t",
        "Unsupervised Cost:",
        "{:.4f}".format(agg_unsupervised_cost_scaled),
        "\t",
        "Validation Accuracy:",
        correct / total,
    )

def l_out_pool(l_in, kernel_size, stride=None, padding=0, dilation=1):
    """
    calculates the pooling output size
    """
    if stride == None:
        stride = kernel_size
    l_out = int(
        np.floor((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    )
    return l_out



def l_out_conv(layer_num, kernel_size, pool=False):
    """
    calculates effective convolution 1D output sizes to define the network
    """
    l_out_list = []
    l_in = constants.SHAPE_OF_ONE_DATA_POINT[1]
    for i in range(layer_num):
        l_out = l_out_conv1d(l_in, kernel_size, stride=2)
        l_out = l_out_conv1d(l_out, kernel_size, stride=2)

        l_out_list.append(l_out)

        if pool:
            pool_size = 3
            l_out = l_out_pool(l_out, pool_size)
            l_out_list.append(l_out)
        l_in = l_out

    # make a copy and reverse for decoder size def

    l_out_list_copy = copy.deepcopy(l_out_list)
    l_out_list.append(32)
    encoder_sizes = l_out_list
    l_out_list_copy.reverse()
    l_out_list_copy.append(constants.SHAPE_OF_ONE_DATA_POINT[1])
    decoder_sizes = l_out_list_copy
    return encoder_sizes, decoder_sizes

def encoder_net_func(num_layer, net_type='cnn'):
    """
    automatically creates the array of layer net type, either cnn or mlp
    :param num_layer: number of layers except the layer softmax layer
    :param net_type: cnn or mlp
    :return: list of net layer types
    """
    ec_funct = []
    for i in range(num_layer):
        ec_funct.append(net_type)
    ec_funct.append('mlp')

    dc_funct = copy.deepcopy(ec_funct)
    dc_funct.reverse()

    return ec_funct, dc_funct


def encoder_train_bn(num_layer):
    """
    a boolean of whether to apply batch norm on data
    taken from the paper as in
    :param num_layer: number of layer
    :return: list of bools, true for last layer
    """
    ec_bn = []
    for i in range(num_layer):
        ec_bn.append(False)
    ec_bn.append(True)
    return ec_bn

def encoder_activation_func(num_layer):
    """
    Generates a list of activation functions
    :param num_layer: numher of layers
    :return: array of activation functions
    """
    ec_funct = []
    for i in range(num_layer):
        ec_funct.append('relu')
    ec_funct.append('softmax')

    return ec_funct

def model_init(args):
    kernel_size = args.kernel
    num_layer = args.num_layer
    encoder_sizes, decoder_sizes = l_out_conv(num_layer, kernel_size, False)
    unsupervised_costs_lambda = [float(x) for x in args.u_costs.split(",")]
    encoder_activations = encoder_activation_func(num_layer)
    encoder_train_bn_scaling = encoder_train_bn(num_layer)

    encoder_layer_type_arr, decoder_layer_type_arr = encoder_net_func(num_layer)
    ladder = Ladder(
        encoder_sizes,
        decoder_sizes,
        encoder_activations,
        encoder_train_bn_scaling,
        args.noise_std,
        args.cuda,
        encoder_layer_type_arr,
        decoder_layer_type_arr,
        kernel_size
    )
    assert len(unsupervised_costs_lambda) == len(decoder_sizes) + 1
    assert len(encoder_sizes) == len(decoder_sizes)
    return ladder


def encoder_forward(ladder, labelled_data, unlabelled_data):
    """
    this aggregates all the submodules of forward pass in the enocders
    Please refer to the refer to the paper for detailed mathematical definitions
    for z, z_tilde
    :param ladder: the ladder model
    :param labelled_data: training data
    :param unlabelled_data: unlabelled data in OMsignal
    :return:
    """
    # do a noisy pass for labelled data
    output_noise_labelled = ladder.forward_encoders_noise(labelled_data)

    # do a noisy pass for unlabelled_data
    output_noise_unlabelled = ladder.forward_encoders_noise(unlabelled_data)
    tilde_z_layers_unlabelled = ladder.get_encoders_tilde_z(reverse=True)

    # do a clean pass for unlabelled data
    output_clean_unlabelled = ladder.forward_encoders_clean(unlabelled_data)

    z_pre_layers_unlabelled = ladder.get_encoders_z_pre(reverse=True)
    z_layers_unlabelled = ladder.get_encoders_z(reverse=True)

    tilde_z_bottom_unlabelled = ladder.get_encoder_tilde_z_bottom()

    res = {"output_noise_labelled": output_noise_labelled,
           "output_noise_unlabelled": output_noise_unlabelled,
           "tilde_z_layers_unlabelled": tilde_z_layers_unlabelled,
           "output_clean_unlabelled": output_clean_unlabelled,
           "z_pre_layers_unlabelled": z_pre_layers_unlabelled,
           "z_layers_unlabelled": z_layers_unlabelled,
           "tilde_z_bottom_unlabelled":tilde_z_bottom_unlabelled}

    return res


def supervised_cost_scale(
    scale, loss_supervised, output_noise_labelled, labelled_target
):
    """
    Ladder network scales each hidden layer loss, usually prioritize the first layer

    :param scale: scale for the final output prediction layer
    :param loss_supervised: total mse loss for supervised
    :param output_noise_labelled: prediction of the noisy endcoder
    :param labelled_target: real target
    :return: total loss of all layers
    """
    cost_supervised = loss_supervised.forward(output_noise_labelled, labelled_target)

    cost_supervised *= scale
    return cost_supervised


def unsupervised_cost_scale(
    unsupervised_costs_lambda,
    z_layers_unlabelled,
    bn_hat_z_layers_unlabelled,
    loss_unsupervised,
):
    """
    Ladder network scales each hidden layer loss, usually prioritize the first layer
    :param unsupervised_costs_lambda: usually 100, 10, 1, 0.1
    :param z_layers_unlabelled: refer to the paper for detailed mathematical definitions
    :param bn_hat_z_layers_unlabelled: refer to the paper for detailed mathematical definitions
    :param loss_unsupervised: total loss of all layers
    :return: total loss of all layers
    """
    cost_unsupervised = 0.0
    for cost_lambda, z, bn_hat_z in zip(
        unsupervised_costs_lambda, z_layers_unlabelled, bn_hat_z_layers_unlabelled
    ):
        c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
        cost_unsupervised += c

    return cost_unsupervised


def get_batch_data(train_loader, device, unlabelled_data, batch_size):
    """
    Loads a batch from loader, use the user_id only
    :param train_loader: data loader torch object
    :param device: gpu or cpu
    :param unlabelled_data: dataset object for unlabelled omsignal
    :param batch_size: size of one batch
    :return: one batch of preprocessed data
    """
    labelled_data, labelled_target = next(iter(train_loader))
    labelled_target = labelled_target[3]
    unlabelled_data = unlabelled_data.to(device)
    labelled_target = labelled_target.to(device=device, dtype=torch.int64)
    labelled_data = labelled_data.to(device)
    labelled_target = labelled_target.squeeze()
    # print("labelled_target", labelled_target.shape)

    labelled_data = labelled_data.view(batch_size, 1, 3750)
    unlabelled_data = unlabelled_data.view(batch_size, 1, 3750)

    labelled_data = pp.Preprocessor().forward(labelled_data)
    unlabelled_data = pp.Preprocessor().forward(unlabelled_data)

    labelled_data = labelled_data.view(batch_size, 1, 3750)
    unlabelled_data = unlabelled_data.view(batch_size, 1, 3750)

    return labelled_data, labelled_target, unlabelled_data


