import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from src.legacy.TeamB1pomt5.code.omsignal.utils.memfile_utils import read_memfile, write_memfile
from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNRegression, CNNClassification
from src.legacy.TeamB1pomt5.code.omsignal.om_networks import CNNRank
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import OM_dataset, Rank_dataset, import_OM, import_train_valid, get_dataloader
from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNRegression, CNNClassification
from src.legacy.TeamB1pomt5.code.omsignal.om_networks import CNNRank
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.legacy.TeamB1pomt5.code.omsignal.utils.rr_stdev import RR_Regressor, make_prediction
from src.legacy.TeamB1pomt5.code.omsignal.utils.pytorch_utils import get_id_mapping, unmap_ids


def eval_model(dataset_file, model_filename):

    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load your best model
    if model_filename:
        model_filename = Path(model_filename)
        print("\nLoading models from", model_filename.absolute())
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        location = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        #Loading ID Classification Model
        ID_CNN = CNNClassification(1876, 32,
                               conv1_num_filters=16, conv2_num_filters=2,
                               conv_ksize=64, num_linear=256, p=0.0, conv_stride=1, conv_padding=4,
                               pool_ksize=5, pool_stride=8, pool_padding=1)
        ID_CNN.to(device)
        ID_CNN.load_state_dict(torch.load('/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt5/model/final_eval_ID_CNN.pt',
                           map_location=location))
        ID_CNN.eval()

        #Loading PR Regression Model
        PR_Model = CNNRegression(3750, conv1_num_filters=2, conv2_num_filters=2,
                           conv_ksize=4, num_linear=128, p=0.8)
        PR_Model.to(device)
        PR_Model.load_state_dict(torch.load('/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt5/model/final_eval_PR_CNN.pt',
                           map_location=location))
        PR_Model.eval()

        #Loading RT Regression Model
        RT_Ranker = CNNRank(3750, conv1_num_filters=16, conv2_num_filters=16,
                        conv_ksize=32, num_linear=256, p=0.8)
        RT_Ranker.to(device)
        RT_Ranker.load_state_dict(torch.load('/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt5/model/final_eval_RT_Ranker.pt',
                              map_location=location))
        RT_Ranker.eval()

        #Loading RR Model
        with open('/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt5/model/rr_stdev.model', 'rb') as fp:
            RR_model = pickle.load(fp)


    #Make id mapping for Classification task
    _, _, y_train, _ = import_train_valid('all', cluster=True)
    mapping = get_id_mapping(y_train[:,3])

    #Load and preprocess the test dataset
    X = read_memfile(dataset_file, shape=(10, 3754), dtype='float32')
    X = X[:,0:3750]
    X = X.reshape(X.shape[0],1,X.shape[1])
    preprocess = Preprocessor()
    preprocess.to(device)
    X = preprocess(torch.from_numpy(X)).numpy()

    # Format PR
    predicted_PR = PR_Model(torch.tensor(X).to(device)).cpu().detach().numpy().astype(np.float32).flatten()

    # Format RT
    predicted_RT_rank = RT_Ranker.Predict_ranking(X, device).astype(np.float32).flatten()

    # Predict RR
    predicted_RR = make_prediction(X,filename='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt5/model/rr_stdev.model').astype(np.float32).flatten()

    # Format ID
    predicted_ID = ID_CNN.Predict_class(X, device).astype(np.int32).flatten()
    predicted_ID = unmap_ids(predicted_ID, mapping)


    y_pred = np.hstack((predicted_PR.reshape((-1, 1)), predicted_RT_rank.reshape((-1, 1)), predicted_RR.reshape((-1, 1)), predicted_ID.reshape((-1, 1))))
    # print("y_pred type", type(y_pred))
    # print("y_pred[0] type", type(y_pred[0]))
    # print("y_pred[0][0] type", type(y_pred[0][0]))
    #
    # print("y_pred", y_pred.shape)
    # print("y_pred", y_pred)
    return y_pred


if __name__ == "__main__":

    ###### DO NOT MODIFY THIS SECTION ######
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--dataset", type=str, default='/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/sample_test.dat')
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
    group_name = "b1pomt5"

    model_filename = "/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1pomt5/model/"
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
