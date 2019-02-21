"""
Author: Y. Violet Guo
Loads the TA'S FINAL PICKLED model
Run a dummy/fake data validation step with TA's model
"""

import sys
import os
import torch
from src.utils import constants
from src.legacy.TABaseline.code.ecgdataset import ECGDataset
import src.legacy.TABaseline.code.baseline_models as models


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""

    for key, module in model.items():
        # if it contains layers let call it recursively to get params and weights
        if show_parameters:
            print(key)
        if show_weights:
            print(module)


def run(model, device):
    print("in runmodel")
    toy_data = ECGDataset(constants.T5_FAKE_VALID_LABELED_DATA)
    toy_loader = torch.utils.data.DataLoader(toy_data)
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(toy_loader):
            print("input {} target {}".format(input, target))
            input = input.to(device)
            # compute output
            output = model(input)
            print("predicted", output)

if __name__ == "__main__":
    learning_rate = 0.001
    momentum = 0.9
    nepoch = 1000
    batch_size = 16

    hidden_size = 16
    dropout = 0.1
    n_layers = 1
    kernel_size = 8
    pool_size = 4

    print(sys.path)

    targets = "pr_mean, rt_mean, rr_stdev, userid"
    target_labels = targets.split(",")

    target_labels = [s.lower().strip() for s in target_labels]

    target_out_size_dict = {"pr_mean": 1, "rt_mean": 1, "rr_stdev": 1, "userid": 32}
    out_size = [target_out_size_dict[a] for a in target_labels]
    model = models.Conv1DBNLinear(
        1, out_size, hidden_size, kernel_size, pool_size, dropout
    )

    device = torch.device('cuda')

    state_dict = torch.load(constants.TA_LEGACY_MODEL)#map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)
    #print(torch_summarize(state_dict, show_weights=False))
    run(model, device)
