from __future__ import print_function

import numpy as np
import argparse
import os
import torch
from torch.autograd import Variable
from torch.optim import Adam
from src.algorithm.encoder import StackedEncoders
from src.algorithm.decoder import StackedDecoders
import src.utils.constants as constants

from src.scripts.unsupervised_pretraining import load_data
from src.legacy.TABaseline.code import Preprocessor as pp


class Ladder(torch.nn.Module):
    def __init__(
        self,
        encoder_sizes,
        decoder_sizes,
        encoder_activations,
        encoder_train_bn_scaling,
        noise_std,
        use_cuda,
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
        )
        self.de = StackedDecoders(decoder_in, decoder_sizes, encoder_in, use_cuda)
        self.bn_image = torch.nn.BatchNorm1d(encoder_in, affine=False)

    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.de.forward(tilde_z_layers, encoder_output, tilde_z_bottom)

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

        # print("type(data)", type(data))


        # ! OLD already a tensor, theorefore commenting out, may need to change for GPU
        # data, target = Variable(data), Variable(target)
        output = ladder.forward_encoders_clean(data)
        # TODO: Do away with the below hack for GPU tensors.
        if args.cuda:
            output = output.cpu()
            target = target.cpu()
        # ! OLD data target already tensor??
        # output = output.data.numpy()
        # target = target.data.numpy()

        # print("type(output)", type(output))
        output = output.detach().numpy()
        # print("type(output)", type(output))
        # print("type(target)", type(target))
        target = target.data.numpy()

        preds = np.argmax(output, axis=1)
        print("*****n preds ******")
        print(preds)
        correct += np.sum(target == preds)
        total += target.shape[0]

    if True: #e % 10:
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


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--u_costs", type=str, default="0.1, 1, 10") # , 0.1, 0.1, 10., 1000.
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
    #TODO: change param reading!
    temp_param = {"batchsize":64}
    batch_size = 64
    train_loader, validation_loader, unlabelled_loader = load_data(
        temp_param, temp_param
    )

    # Configure the Ladder
    starter_lr = 0.002
    encoder_sizes = [1024, 32] # , 2048, 1024, 32]  # 32 or 35
    decoder_sizes = [1024, constants.SHAPE_OF_ONE_DATA_POINT[1]] # [32, 1024, 2048, constants.SHAPE_OF_ONE_DATA_POINT[1]]
    unsupervised_costs_lambda = [float(x) for x in args.u_costs.split(",")]
    encoder_activations =  ["relu", "softmax"] # ["relu", "relu", "relu", "softmax"]
    encoder_train_bn_scaling = [False, True] #[False, False, False, True]
    ladder = Ladder(
        encoder_sizes,
        decoder_sizes,
        encoder_activations,
        encoder_train_bn_scaling,
        noise_std,
        args.cuda,
    )
    optimizer = Adam(ladder.parameters(), lr=starter_lr)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()

    if args.cuda:
        ladder.cuda()

    assert len(unsupervised_costs_lambda) == len(decoder_sizes) + 1
    assert len(encoder_sizes) == len(decoder_sizes)

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
    use_gpu = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_gpu else "cpu")

    for e in range(epochs):
        agg_cost = 0.0
        agg_supervised_cost = 0.0
        agg_unsupervised_cost = 0.0
        num_batches = 0
        ladder.train()

        for batch_idx, (unlabelled_data, _) in enumerate(unlabelled_loader):

            # TODO: Verify whether labelled examples are used for calculating unsupervised loss.
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

            labelled_data = labelled_data.view(batch_size, 3750)
            unlabelled_data = unlabelled_data.view(batch_size, 3750)

            optimizer.zero_grad()

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

            # pass through decoders
            hat_z_layers_unlabelled = ladder.forward_decoders(
                tilde_z_layers_unlabelled,
                output_noise_unlabelled,
                tilde_z_bottom_unlabelled,
            )

            z_pre_layers_unlabelled.append(unlabelled_data)
            z_layers_unlabelled.append(unlabelled_data)

            # TODO: Verify if you have to batch-normalize the bottom-most layer also
            # batch normalize using mean, var of z_pre
            bn_hat_z_layers_unlabelled = ladder.decoder_bn_hat_z_layers(
                hat_z_layers_unlabelled, z_pre_layers_unlabelled
            )

            # calculate costs
            cost_supervised = loss_supervised.forward(
                output_noise_labelled, labelled_target
            )
            cost_unsupervised = 0.0
            assert len(z_layers_unlabelled) == len(bn_hat_z_layers_unlabelled)
            for cost_lambda, z, bn_hat_z in zip(
                unsupervised_costs_lambda,
                z_layers_unlabelled,
                bn_hat_z_layers_unlabelled,
            ):
                c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
                cost_unsupervised += c

            # backprop
            cost = cost_supervised + cost_unsupervised
            cost.backward()
            optimizer.step()

            # ! OLD # agg_cost += cost.data[0]
            # agg_supervised_cost += cost_supervised.data[0]
            # agg_unsupervised_cost += cost_unsupervised.data[0]
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
