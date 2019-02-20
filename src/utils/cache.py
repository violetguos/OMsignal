import os


"""Util functions to support saving models, including arugments, results, etc"""


def save_args(args):
    # Save argparse arguments to a text file for future reference
    os.makedirs(args.save_directory, exist_ok=True)
    with open(os.path.join(args.save_directory, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


