import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import os
import sys
from src.utils.os_helper import write_memfile
sys.path.append(os.path.abspath(os.path.join('..')))



def eval_model(dataset_file, model_filename):
    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load your best model
    pass





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
