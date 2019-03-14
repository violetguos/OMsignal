from algorithm.ladder.ladder import (
    model_init,
    evaluate_performance,
    layer_plot,
    supervised_cost_scale,
    unsupervised_cost_scale,
    get_batch_data,
    encoder_forward
)
from src.scripts.unsupervised_pretraining import load_data
import torch
import numpy as np
import argparse
from torch.optim import Adam


def print_net(ladder, unsupervised_costs_lambda):
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

def args_setup():
    # command line arguments
    parser = argparse.ArgumentParser(description="Parser for Ladder network")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel", type=int, default=8)
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument(
        "--u_costs", type=str, default="1., 1., 1."
    )  # , 0.1, 0.1, 10., 1000.
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    return args

def main():
    args = args_setup()
    epochs = args.epochs
    seed = args.seed
    if args.cuda and not torch.cuda.is_available():
        print("WARNING: torch.cuda not available, using CPU.\n")
        args.cuda = False

    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    print("Loading Data")
    batch_size = args.batch
    batch_config = {"batchsize": batch_size}
    train_loader, validation_loader, unlabelled_loader = load_data(
        batch_config, batch_config
    )

    # layerwise scales for the MSE loss
    unsupervised_costs_lambda = [float(x) for x in args.u_costs.split(",")]

    # declare the model
    ladder = model_init(args)
    if args.cuda:
        ladder.cuda()

    print_net(ladder, unsupervised_costs_lambda)

    # configure the optimizer
    starter_lr = 0.1

    optimizer = Adam(ladder.parameters(), lr=starter_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3, eta_min=1e-5)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()


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


            bn_hat_z_layers_unlabelled = ladder.decoder_bn_hat_z_layers(
                hat_z_layers_unlabelled, encoder_res["z_pre_layers_unlabelled"]
            )
            assert len(encoder_res["z_layers_unlabelled"]) == len(
                bn_hat_z_layers_unlabelled
            )

            # calculate costs with layerwise scaling
            scale = 10
            cost_supervised = supervised_cost_scale(
                scale,
                loss_supervised,
                encoder_res["output_noise_labelled"],
                labelled_target,
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
            scheduler.step(cost)
            agg_cost += cost.data
            agg_supervised_cost += cost_supervised.data
            agg_unsupervised_cost += cost_unsupervised.data
            num_batches += 1

        # plot at the end of every epooch
        plot_dict = {
            "hat_z_layers_unlabelled": hat_z_layers_unlabelled[-1][0],
            "unlabelled_data": unlabelled_data[0],
        }
        layer_plot(plot_dict, title="ladder {}".format(e), fig="ladder_{}".format(e))

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
