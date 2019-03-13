import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import os
import sys
from src.utils.os_helper import write_memfile
sys.path.append(os.path.abspath(os.path.join('..')))

# Block 2 Team 1 custom imports
import src.legacy.TABaseline.code.ecgdataset as ecgdataset
from torch.utils.data import DataLoader
from src.utils import constants


def eval_model(dataset_file, model_filename):

    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.
    '''
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load your best model
    if model_filename:
        model_filename = Path(model_filename)
        print("\nLoading model from", model_filename.absolute())
        model = torch.load(model_filename, map_location=device)

        # Load a multitask model


    if model:
        targets = constants.TARGETS
        # according to our config files

        batch_size = 16
        # load data
        test_dataset = ecgdataset.ECGDataset(
            dataset_file, False, target=targets
        )
        test_loader = DataLoader(
            test_dataset, batch_size, shuffle=False, num_workers=1
        )
        model.eval()
        # record results in a list, 
        y_pred_list = []
        for x, y in test_loader:
            x = x.to(device)

            outputs = model(x)
            if device == 'cpu':
                y_pred_list.append(outputs.data.numpy())
            else:
                y_pred_list.append(outputs.cpu().data.numpy())
        # concate a list of numpy arrays
        y_pred = np.concatenate(y_pred_list)


    else:

        print("\nYou did not specify a model, generating dummy data instead!")
        n_classes = 32
        num_data = 10

        y_pred = np.concatenate(
            [np.random.rand(num_data, 3),
             np.random.randint(0, n_classes, (num_data, 1))
             ], axis=1
        ).astype(np.float32)

    return y_pred




if __name__ == "__main__":
    ###### DO NOT MODIFY THIS SECTION ######
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='')
    # dataset_dir will be the absolute path to the dataset to be used for
    # evaluation.

    parser.add_argument("--results_dir", type=str, default='')
    # results_dir will be the absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    dataset_file = args.dataset
    results_dir = args.results_dir
    #########################################

    ###### MODIFY THIS SECTION ######
    # Put your group name here
    group_name = "b2pomt1"

    model_filename = "/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt1/model/"
    # model_filename should be the absolute path on shared disk to your
    # best model. You need to ensure that they are available to evaluators on
    # Helios.

    #################################

    ###### DO NOT MODIFY THIS SECTION ######
    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_file, model_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 2, "Make sure ndim=2 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    write_memfile(y_pred, results_fname)
#########################################
