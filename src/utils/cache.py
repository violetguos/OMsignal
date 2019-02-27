from datetime import datetime
from pytz import timezone
import os
from src.utils.constants import SAVE_MODEL_PATH
import torch

"""Util functions to support saving models, including arugments, results, etc"""


def save_args(args):
    # Save argparse arguments to a text file for future reference
    os.makedirs(args.save_directory, exist_ok=True)
    with open(os.path.join(args.save_directory, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


class ModelCache:
    """create a new object for each experiment"""

    def __init__(self, _prefix="model_cache", _mode="train", _suffix=None):
        self.prefix = _prefix
        self.mode = _mode
        self.time_freeze = self.time_gen()
        if _suffix:
            self.dir = os.path.join(SAVE_MODEL_PATH, self.time_freeze + _suffix)
        else:
            self.dir = os.path.join(SAVE_MODEL_PATH, self.time_freeze)

    def time_gen(self):
        fmt = "%Y_%m_%d_%H_%M_%S_"
        now_time = datetime.now(timezone("US/Eastern"))
        return now_time.strftime(fmt)

    def save(self, model, epoch, verbose=False):
        model_dir = os.path.join(self.dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        fname = self.prefix + "_" + "epoch_" + str(epoch)
        fname = fname + ".pt"

        fp = os.path.join(model_dir, fname)
        if verbose:
            print("saving torch model")
            print("model_dir", model_dir)
            print("fname", fname)
            print("fp", fp)
        torch.save(model.state_dict(), fp)

    def log(self, epoch, loss, verbose=False):
        """

        :param loss: list of errors

        :param epoch: number of epoch
        :return: a log file
        """
        log_dir = os.path.join(self.dir, "history")
        os.makedirs(log_dir, exist_ok=True)
        fname = self.prefix + "_" + self.mode
        fname = fname + ".txt"
        fp = os.path.join(log_dir, fname)
        if verbose:
            print("saving log errors")
            print("log_dir", log_dir)
            print("fname", fname)
            print("fp", fp)

        with open(fp, "a+") as f:
            line = "{},{}\n".format(epoch, loss)
            f.write(line)
