from __future__ import print_function

import numpy as np
import argparse
import os
import torch
import copy
from torch.autograd import Variable
from torch.optim import Adam
from src.algorithm.encoder import StackedEncoders, l_out_conv1d
from src.algorithm.decoder import StackedDecoders, inv_l_out
import src.utils.constants as constants

from src.scripts.unsupervised_pretraining import load_data
from src.legacy.TABaseline.code import Preprocessor as pp


def l_out_pool(l_in, kernel_size, stride=None, padding=0, dilation=1):
    """
    calculates the pooling output size according to the official pytorch doc
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
        l_out = l_out_conv1d(l_in, kernel_size)
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
        )
        self.de = StackedDecoders(
            decoder_in, decoder_sizes, encoder_in, use_cuda, decoder_layer_type_arr
        )
        self.bn_image = torch.nn.BatchNorm1d(encoder_in, affine=False)

    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.de.forward(
            tilde_z_layers, encoder_output[0], tilde_z_bottom, encoder_output[1]
        )

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

        # extra _ from the forward function was intended to be the indices of MaxUnpool
        output, _ = ladder.forward_encoders_clean(data)
        # TODO: don't think we need to convert back to CPU, just need to detach
        if args.cuda:
            output = output.cpu()
            target = target.cpu()

        output = output.detach().numpy()
        target = target.data.numpy()

        preds = np.argmax(output, axis=1)

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


def model_init(args):
    kernel_size = 8
    num_layer = 3
    encoder_sizes, decoder_sizes = l_out_conv(num_layer, kernel_size, False)
    print("encoder", encoder_sizes)
    print("decoder", decoder_sizes)
    unsupervised_costs_lambda = [float(x) for x in args.u_costs.split(",")]
    encoder_activations = ["relu", "relu", "relu", "relu", "softmax"]
    encoder_train_bn_scaling = [False, False, False, False, True]

    encoder_layer_type_arr = ["cnn", "cnn", "cnn", "mlp"]
    decoder_layer_type_arr = ["mlp", "cnn", "cnn", "cnn"]
    ladder = Ladder(
        encoder_sizes,
        decoder_sizes,
        encoder_activations,
        encoder_train_bn_scaling,
        args.noise_std,
        args.cuda,
        encoder_layer_type_arr,
        decoder_layer_type_arr,
    )
    assert len(unsupervised_costs_lambda) == len(decoder_sizes) + 1
    assert len(encoder_sizes) == len(decoder_sizes)
    return ladder


def encoder_forward(ladder, labelled_data, unlabelled_data):
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
    cost_supervised = loss_supervised.forward(output_noise_labelled[0], labelled_target)

    cost_supervised *= scale
    return cost_supervised


def unsupervised_cost_scale(
    unsupervised_costs_lambda,
    z_layers_unlabelled,
    bn_hat_z_layers_unlabelled,
    loss_unsupervised,
):
    cost_unsupervised = 0.0
    for cost_lambda, z, bn_hat_z in zip(
        unsupervised_costs_lambda, z_layers_unlabelled, bn_hat_z_layers_unlabelled
    ):
        c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
        cost_unsupervised += c

    return cost_unsupervised


def get_batch_data(train_loader, device, unlabelled_data, batch_size):
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

    # TODO: add a switch for MLP vs CNN
    labelled_data = labelled_data.view(batch_size, 1, 3750)
    unlabelled_data = unlabelled_data.view(batch_size, 1, 3750)

    # labelled_data = labelled_data.view(batch_size, 3750)
    # unlabelled_data = unlabelled_data.view(batch_size, 3750)

    return labelled_data, labelled_target, unlabelled_data


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--u_costs", type=str, default="0.1, 1,  1, 1, 10"
    )  # , 0.1, 0.1, 10., 1000.
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--decay_epoch", type=int, default=15)
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs
    noise_std = args.noise_std
    seed = args.seed
    decay_epoch = args.decay_epoch
    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.\n")
        args.cuda = False

    print("=====================")
    print("BATCH SIZE:", batch_size)
    print("EPOCHS:", epochs)
    print("RANDOM SEED:", args.seed)
    print("NOISE STD:", noise_std)
    print("LR DECAY EPOCH:", decay_epoch)
    print("CUDA:", args.cuda)
    print("=====================\n")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    print("Loading Data")
    # TODO: change param reading!
    temp_param = {"batchsize": 16}
    batch_size = 16

    train_loader, validation_loader, unlabelled_loader = load_data(
        temp_param, temp_param
    )

    # Configure the Ladder
    # TODO: variable kernel size
    # TODO: variable channel size
    # TODO: change param reading!

    # keep this
    unsupervised_costs_lambda = [float(x) for x in args.u_costs.split(",")]

    # declare the model
    ladder = model_init(args)
    if args.cuda:
        ladder.cuda()

    # configure the optimizer
    starter_lr = 0.02
    optimizer = Adam(ladder.parameters(), lr=starter_lr)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()

    print("")
    print("========NETWORK=======")
    print(ladder)
    print("======================")

    print("")
    print("==UNSUPERVISED-COSTS==")
    print(unsupervised_costs_lambda)

    print("")
    print("=====================")
    print("TRAINING\n")

    # TODO: Add learning rate scheduler

    # TODO: make this less repritive, set global
    use_gpu = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_gpu else "cpu")

    for e in range(epochs):
        agg_cost = 0.0
        agg_supervised_cost = 0.0
        agg_unsupervised_cost = 0.0
        num_batches = 0
        ladder.train()

        for batch_idx, (unlabelled_data, _) in enumerate(unlabelled_loader):
            labelled_data, labelled_target, unlabelled_data = get_batch_data(
                train_loader, device, unlabelled_data, batch_size
            )

            optimizer.zero_grad()

            # forward pass in both noisy and clean encoders
            encoder_res = encoder_forward(ladder, labelled_data, unlabelled_data)

            # pass through decoders
            hat_z_layers_unlabelled = ladder.forward_decoders(
                encoder_res["tilde_z_layers_unlabelled"],
                encoder_res["output_noise_unlabelled"],
                encoder_res["tilde_z_bottom_unlabelled"],
            )

            encoder_res["z_pre_layers_unlabelled"].append(unlabelled_data)
            encoder_res["z_layers_unlabelled"].append(unlabelled_data)

            # TODO: Verify if you have to batch-normalize the bottom-most layer also
            # batch normalize using mean, var of z_pre
            # inputs are type list
            bn_hat_z_layers_unlabelled = ladder.decoder_bn_hat_z_layers(
                hat_z_layers_unlabelled, encoder_res["z_pre_layers_unlabelled"]
            )
            assert len(encoder_res["z_layers_unlabelled"]) == len(bn_hat_z_layers_unlabelled)

            # calculate costs
            # TODO hyperparam scale
            scale = 10
            cost_supervised = supervised_cost_scale(
                scale, loss_supervised, encoder_res["output_noise_labelled"], labelled_target
            )

            # unsupercised cost and scaling each layer
            cost_unsupervised = unsupervised_cost_scale(
                unsupervised_costs_lambda,
                encoder_res["z_layers_unlabelled"],
                bn_hat_z_layers_unlabelled,
                loss_unsupervised,
            )

            # backprop
            cost = cost_supervised + cost_unsupervised
            cost.backward()
            optimizer.step()

            agg_cost += cost.data
            agg_supervised_cost += cost_supervised.data
            agg_unsupervised_cost += cost_unsupervised.data
            num_batches += 1

        # Evaluation
        ladder.eval()
        evaluate_performance(
            ladder,
            validation_loader,
            e,
            agg_cost / num_batches,
            agg_supervised_cost / num_batches,
            agg_unsupervised_cost / num_batches,
            args,
        )
        # back to train mode

        ladder.train()
    print("=====================\n")
    print("Done :)")


if __name__ == "__main__":
    main()
