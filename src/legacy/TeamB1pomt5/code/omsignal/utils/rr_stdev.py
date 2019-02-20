import numpy as np
import torch
import os
import sys
import argparse
import pickle
import datetime
from scipy.stats import zscore
from scipy.stats import kendalltau
from sklearn.tree import DecisionTreeRegressor as dtr
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from omsignal.utils.preprocessor import Preprocessor
try:
    from omsignal.utils.dataloader_utils import import_train_valid
except:
    pass
from config import MODELS_DIR

class RR_Regressor(object):
    def __init__(self, threshold, distance, z_threshold):
        self.threshold = threshold
        self.distance = distance
        self.z_threshold = z_threshold

    
def finite_derivative(sequence):
    derivative = np.zeros(sequence.shape)

    for i in range(len(sequence)):
        if i == 0:
            derivative[i] = sequence[i+1]-sequence[i]
        if i == len(sequence)-1:
            derivative[i] = sequence[i]-sequence[i-1]
        else:
            derivative[i] = (sequence[i+1]-sequence[i-1])
    return derivative

def find_R_peaks(sequence, threshold, distance):
    nb_datapoints = len(sequence)
    position = []
    for i in range(1,nb_datapoints-1):
        if sequence[i] > sequence[i-1] and sequence[i] > sequence[i+1] and sequence[i] > threshold:
            if i < distance:
                if sequence[i] == np.max(sequence[0:i+distance]):
                    position = np.append(position, i)
            elif i > len(sequence) - distance:
                if sequence[i] == np.max(sequence[i-distance:nb_datapoints]):
                    position = np.append(position, i)
            else:
                if sequence[i] == np.max(sequence[i-distance:i+distance]):
                    position = np.append(position, i)
    if len(position) < 3:
        #If not enough peaks are found, a dummy sequence with poor performance is returned
        return [0, 50, 2000]
    else:
        return position.astype(int)

def remove_outliers(diffs, z_threshold):
    pruned_diffs = diffs
    z_score = np.abs(zscore(diffs))
    for i in range(len(z_score)):
        if z_score[i] > z_threshold:
            pruned_diffs[i] = 0
    return pruned_diffs[np.nonzero(pruned_diffs)]


def std_preprocess(cluster):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, X_valid, y_train, y_valid = import_train_valid('RR_stdevs', cluster = cluster)

    preprocess = Preprocessor()
    preprocess.to(device)

    X_train_preprocessed = preprocess(torch.from_numpy(X_train)).numpy()
    X_valid_preprocessed = preprocess(torch.from_numpy(X_valid)).numpy()
    
    X_train_preprocessed = X_train_preprocessed[:,0,:]
    X_valid_preprocessed = X_valid_preprocessed[:,0,:]

    return X_train_preprocessed, X_valid_preprocessed, y_train, y_valid

def compute_std(sequence, threshold, distance, z_threshold):
    R_Peaks_idx = find_R_peaks(sequence, threshold, distance)
    R_Peaks_diff = np.diff(R_Peaks_idx)
    pruned_diffs = remove_outliers(R_Peaks_diff, z_threshold)
    if len(pruned_diffs) < 1:
        #If every peak is pruned, a dummy std with poor performance is returned
        return 200
    else:
        return np.nanstd(pruned_diffs)

def batch_compute_std(batch, threshold, distance, z_threshold):
    stdevs = np.zeros(batch.shape[0])
    for i, item in enumerate(batch):
        # Must index at [0] - due to dataloader weirdness
        stdev = compute_std(item[0], threshold, distance, z_threshold)
        stdevs[i] = stdev
    return stdevs

def compute_performance(x,y,threshold,distance,z_threshold):
    y_predict = np.zeros(len(y))
    kTau_score = 0
    for i in range(len(y)):
        y_predict[i] = compute_std(x[i],threshold,distance,z_threshold)
    
    kTau_score, _ = kendalltau(y_predict, y)
    return y_predict, kTau_score

def make_prediction(X, filename = None):
    threshold, distance, z_threshold = load_model(filename=filename)
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        y_predict[i] = compute_std(X[i].flatten(),threshold,distance,z_threshold)
    return y_predict


def hyperparam_tuning(X_train_preprocessed, X_valid_preprocessed, y_train, y_valid,params):
    '''
    params[0] to [8]: [min threshold, max threshold, min distance, max distance,\
        min z_threshold, max z_threshold, saved threshold, saved distance, saved z_thresold]
    '''
    #load from best model
    tuned_threshold = params[6]
    tuned_distance = params[7]
    tuned_z_threshold = params[8]
    _, tuned_train_score = compute_performance(X_train_preprocessed,y_train,tuned_threshold,tuned_distance,tuned_z_threshold)
    _, tuned_valid_score = compute_performance(X_valid_preprocessed,y_valid,tuned_threshold,tuned_distance,tuned_z_threshold)

    for distance in range(params[2],params[3]):
        for threshold in range(params[0],params[1]):
            for z_threshold in range(params[4],params[5]):
                _, train_score = compute_performance(X_train_preprocessed,y_train,threshold/10,distance,z_threshold/10)
                _, valid_score = compute_performance(X_valid_preprocessed,y_valid,threshold/10,distance,z_threshold/10)
                print("Train score: {0:8.3f}  Valid score: {1:8.3f} - Thresh: {2:2.1f} - Dist: {3:2.0f} - Z_Thresh: {4:2.1f}".format(train_score,valid_score,threshold/10,distance,z_threshold/10))
                if train_score > tuned_train_score:
                    tuned_threshold = threshold/10
                    tuned_distance = distance
                    tuned_z_threshold = z_threshold/10
                    tuned_train_score = train_score
                    tuned_valid_score = valid_score
            save_model(tuned_threshold,tuned_distance,tuned_z_threshold)
            print("Saved model kendall Score on training and validation: {0:5.3f} - {1:5.3f}".format(tuned_train_score, tuned_valid_score))
        logs_filename = os.path.join(MODELS_DIR, 'last_RR_model.txt')
        np.savetxt(logs_filename, [threshold,distance,z_threshold,train_score,valid_score], fmt='%8.3f', header = 'Threshold - Distance - Z_Threshold - Train score - Valid score\n')

def save_model(threshold, distance, z_threshold):
    model = RR_Regressor(threshold, distance, z_threshold)
    filename = os.path.join(MODELS_DIR, 'rr_stdev.model')
    file_model = open(filename, 'wb')
    pickle.dump(model, file_model)
    print('Saved RR_stdev Regression Model. Threshold: {0:2.1f} - Distance: {1:2.0f} - Z_Threshold: {2:2.1f}'.format(threshold, distance, z_threshold))

def load_model(filename=None):
    if filename == None:
        filename = os.path.join(MODELS_DIR, 'rr_stdev.model')
    
    with open(filename, 'rb') as fp:
        model = pickle.load(fp)
    #print('Loaded RR_stdev Regression Model. Threshold: {0:2.1f} - Distance: {1:2.0f} - Z_Threshold: {2:2.1f}'.format(model.threshold, model.distance, model.z_threshold))
    return model.threshold, model.distance, model.z_threshold

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Generate line graph visualizations of the data.')
    parser.add_argument('mode', help='f: find the best hyperparameters from the training set.\
        p: evaluate the performance of the current saved model.', type=str)
    parser.add_argument('--thr', nargs=2, help='Finetuning range for threshold. \
        Example use: --thr 3 40 / Default values: 5 50 / Automatic resizing of 0.1')
    parser.add_argument('--dis', nargs=2, help='Finetuning range for distance. \
        Example use: --dis 5 20  Default values: 1 25')
    parser.add_argument('--zth', nargs=2, help='Finetuning range for z_threshold \
        Example use: --zth 5 15  Default values: 8 20 / Automatic resizing of 0.1')
    parser.add_argument('--cluster', help='Flag for running on the Helios cluster. \
        If flagged, will use real data; otherwise, will use dummy data.', action='store_true')
    args = parser.parse_args()
   
    threshold, distance, z_threshold = load_model()

    if args.mode == 'f':
        params = [5,50,1,25,8,20,threshold,distance,z_threshold]
        if args.thr:
            if int(args.thr[0]) < int(args.thr[1]) and int(args.thr[0]) > 0:
                params[0] = int(args.thr[0])
                params[1] = int(args.thr[1])
        if args.dis:
            if int(args.dis[0]) < int(args.dis[1]) and int(args.dis[0]) > 0:
                params[2] = int(args.dis[0])
                params[3] = int(args.dis[1])
        if args.zth:
            if int(args.zth[0]) < int(args.zth[1]) and int(args.zth[0]) > 0:
                params[4] = int(args.zth[0])
                params[5] = int(args.zth[1])
        X_train_preprocessed, X_valid_preprocessed, y_train, y_valid = std_preprocess(cluster=args.cluster)
        hyperparam_tuning(X_train_preprocessed, X_valid_preprocessed, y_train, y_valid,params)
    elif args.mode == 'p':
        X_train_preprocessed, X_valid_preprocessed, y_train, y_valid = std_preprocess(cluster=args.cluster)
        _, score = compute_performance(X_valid_preprocessed,y_valid,threshold,distance,z_threshold)
        print("Score of current best model on validation: {0:5.3f}".format(score))
