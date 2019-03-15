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
from src.data.unlabelled_data import UnlabelledDataset
from torch.utils.data import DataLoader
from src.utils import constants
from src.algorithm.CNN_multitask_semisupervised import Conv1DBNLinear
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import unmap_ids, get_id_mapping
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import import_train_valid



def eval_model(dataset_file, model_filename):

    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.
    '''
    model = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    location = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    targets = constants.TARGETS
    target_out_size_dict = {"pr_mean": 1, "rt_mean": 1, "rr_stdev": 1, "userid": 32}
    target_labels = targets.split(",")
    target_labels = [s.lower().strip() for s in target_labels]


    hidden_size = 16
    kernel_size = 8
    pool_size = 4
    dropout = 0.1

    # Load your best model
    if model_filename:
        model_filename = Path(model_filename)
        print("\nLoading model from", model_filename.absolute())
        # define the model
        # Model initialization
        target_labels = targets.split(",")
        target_labels = [s.lower().strip() for s in target_labels]
        if len(target_labels) == 1:
            out_size = target_out_size_dict[target_labels[0]]
        else:
            out_size = [target_out_size_dict[a] for a in target_labels]

        # Load a multitask model
        model = Conv1DBNLinear(
        1, out_size, hidden_size, kernel_size, pool_size, dropout
    )
        # model.to_device(device)
        model.load_state_dict(torch.load('/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt1/model/b2pomt1_final_model.pt',
                           map_location=location))
        model.to(device)

    if model:
        # according to our config files
        batch_size = 10
        # load data

        test_dataset = UnlabelledDataset(
            dataset_file
        )
        test_loader = DataLoader(
            test_dataset, batch_size, shuffle=False, num_workers=1
        )

        score_param_index = [
            None if target_labels.count("pr_mean") == 0 else target_labels.index("pr_mean"),
            None if target_labels.count("rt_mean") == 0 else target_labels.index("rt_mean"),
            None
            if target_labels.count("rr_stdev") == 0
            else target_labels.index("rr_stdev"),
            None if target_labels.count("userid") == 0 else target_labels.index("userid"),
        ]

        prMean_pred  = None
        rtMean_pred  = None
        rrStd_pred  = None
        ecgId_pred = None


        model.eval()

        for x, _ in test_loader:
            x = x.to(device)
            x = x.view(batch_size, 1, 3750)

            outputs = model(x)
            if score_param_index[0] is not None:
                i = score_param_index[0]
                if prMean_pred is None:
                    prMean_pred = outputs[i].view(-1).tolist()
                else:
                    prMean_pred.extend(outputs[i].view(-1).tolist())
            if score_param_index[1] is not None:
                i = score_param_index[1]
                if rtMean_pred is None:
                    rtMean_pred = outputs[i].view(-1).tolist()
                else:
                    rtMean_pred.extend(outputs[i].view(-1).tolist())
            if score_param_index[2] is not None:
                i = score_param_index[2]
                if rrStd_pred is None:
                    rrStd_pred = outputs[i].view(-1).tolist()
                else:
                    rrStd_pred.extend(outputs[i].view(-1).tolist())
            if score_param_index[3] is not None:
                i = score_param_index[3]
                _, pred_classes = torch.max(outputs[i], dim=1)
                if ecgId_pred is None:
                    ecgId_pred = pred_classes.view(-1).tolist()
                else:
                    ecgId_pred.extend(pred_classes.view(-1).tolist())
            # print("type prMean_pred[0]", type(prMean_pred[0]))

        # metrics
        prMean_pred = None if prMean_pred is None else np.array(
            prMean_pred, dtype=np.float32)
        # print("prMean_pred", prMean_pred.shape)
        # print("type prMean_pred[0]", type(prMean_pred[0]))
        rtMean_pred = None if rtMean_pred is None else np.array(
            rtMean_pred, dtype=np.float32)

        rrStd_pred = None if rrStd_pred is None else np.array(
            rrStd_pred, dtype=np.float32)

        ecgId_pred = None if ecgId_pred is None else np.array(
            ecgId_pred, dtype=np.int32)

        # Make id mapping for Classification task
        _, _, y_train, _ = import_train_valid('all', cluster=True)
        mapping = get_id_mapping(y_train[:, 3])
        # map back
        ecgId_pred = unmap_ids(ecgId_pred, mapping)

        y_pred = np.hstack((prMean_pred.reshape((-1, 1)), rtMean_pred.reshape((-1, 1)),
                            rrStd_pred.reshape((-1, 1)), ecgId_pred.reshape((-1, 1))))


        y_pred = y_pred.astype(np.float32)

    else:

        print("\nYou did not specify a model, generating dummy data instead!")
        n_classes = 32
        num_data = 10

        y_pred = np.concatenate(
            [np.random.rand(num_data, 3),
             np.random.randint(0, n_classes, (num_data, 1))
             ], axis=1
        ).astype(np.float32)

    print("y_pred type", type(y_pred))
    print("y_pred[0] type", type(y_pred[0]))
    print("y_pred[0][0] type", type(y_pred[0][0]))

    print("y_pred", y_pred.shape)
    print("y_pred", y_pred)

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
