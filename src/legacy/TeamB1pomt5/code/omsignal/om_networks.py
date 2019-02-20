import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

from src.legacy.TeamB1pomt5.code.omsignal.base_networks import CNNRegression, CNNClassification
from src.legacy.TeamB1pomt5.code.omsignal.utils.fft_utils import make_tensor_fft
from functools import cmp_to_key

def custom_sort_single(a, b, model, device):
    comp_ab = torch.Tensor(np.vstack((a[1],b[1])).reshape(1,1,2,3750)).float().to(device)
    vote = torch.sigmoid(model.forward(comp_ab)).item() < 0.5
    return 0.5 - vote

class CNNRank(nn.Module):
    '''
    CNN that learns to rank pairs - doesn't care about actual values predicted,
    just their order relative to each other
    '''
    def __init__(self, input_size, conv1_num_filters=2, conv2_num_filters=1,
                 conv_ksize=10, conv_stride=1, conv_padding=4,
                 pool_ksize=5, pool_stride=8, pool_padding=1,
                 num_linear=100, p=0.5):
        super(CNNRank, self).__init__()
        self.CNN = CNNRegression(input_size,
                                 conv1_num_filters=conv1_num_filters, conv2_num_filters=conv2_num_filters,
                                 conv_ksize=conv_ksize, conv_stride=conv_stride, conv_padding=conv_padding,
                                 pool_ksize=pool_ksize, pool_stride=pool_stride, pool_padding=pool_padding,
                                 num_linear=num_linear, p=p)

    def forward(self, x):
        seq1 = x[:,:,0,:].view((x.size(0),1,3750))
        seq2 = x[:,:,1,:].view((x.size(0),1,3750))

        # Ensure that seq1 > seq2 is equivalent to seq2 < seq1
        result = self.CNN(seq1) - self.CNN(seq2)
        return result


    def Predict_ranking(self, X, device):
        self.eval()
        arr_copy = np.copy(X)
        
        comp = lambda a,b: custom_sort_single(a, b, self, device)
        X_sorted = sorted(((idx, X[idx]) for idx in range(arr_copy.shape[0])), key=cmp_to_key(comp))
        #old_idx_sorted returns the original indexes of the sorted sequences 
        old_sorted = list(list(zip(*X_sorted))[0])
        predicted_rank = np.zeros((arr_copy.shape[0],1))
        for i in range(arr_copy.shape[0]):
            predicted_rank[old_sorted[i]] = i
        return predicted_rank

"""
old Network modules, we don't use these classes but might be useful to next teams

class MultitaskMLP(nn.Module):
    '''
    Version of MLP with multitask output
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultitaskMLP, self).__init__()
        # Regression
        self.pr_regressor = MLPRegression(input_size, hidden_size)
        self.rt_regressor = MLPRegression(input_size, hidden_size)
        self.rr_regressor = MLPRegression(input_size, hidden_size)
        # Classification
        self.id_classifier = MLPClassification(input_size, hidden_size, num_classes)
    
    def forward(self, x):
        pr = self.pr_regressor(x)
        rt = self.rt_regressor(x)
        rr = self.rr_regressor(x)
        i = self.id_classifier(x)

        return pr, rt, rr, i

class MultitaskCNN(nn.Module):
    '''
    Version of CNN with multitask output.
    '''
    def __init__(self, reg_input_size, class_input_size, num_classes):
        super(MultitaskCNN, self).__init__()
        # Regression
        #self.pr_regressor = CNNRegression(reg_input_size)
        self.rt_regressor = CNNRegression(reg_input_size)
        #self.rr_regressor = CNNRegression(reg_input_size, num_linear=500)
        # Classification
        self.id_classifier = CNNClassification(class_input_size, num_classes)

    def forward(self, x):
        #pr = self.pr_regressor(x)
        rt = self.rt_regressor(x)
        #rr = self.rr_regressor(x)

        # Apply fft to classification input
        i = make_tensor_fft(x)
        i = self.id_classifier(i)

        #return pr, rt, rr, i
        #return pr, rt, i
        return rt, i

class HybridMLPCNN(nn.Module):
    '''
    Uses MLP for regression and CNN for classification.
    '''
    def __init__(self, reg_input_size, class_input_size, hidden_size, num_classes):
        super(HybridMLPCNN, self).__init__()
        # Regression
        self.pr_regressor = MLPRegression(reg_input_size, hidden_size)
        self.rt_regressor = MLPRegression(reg_input_size, hidden_size)
        #self.rr_regressor = MLPRegression(reg_input_size, hidden_size)
        # Classification
        self.id_classifier = CNNClassification(class_input_size, num_classes)

    def forward(self, x):
        pr = self.pr_regressor(x)
        rt = self.rt_regressor(x)
        rr = self.rr_regressor(x)

        # Apply fft to classification input
        i = make_tensor_fft(x)
        i = self.id_classifier(i)

        return pr, rt, rr, i

class HybridCNNMLP(nn.Module):
    '''
    The reverse of the preceding model - 
    Uses CNN for regression and MLP for classification.
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        super(HybridCNNMLP, self).__init__()
        # Regression
        self.pr_regressor = CNNRegression(input_size)
        self.rt_regressor = CNNRegression(input_size)
        self.rr_regressor = CNNRegression(input_size)
        # Classification
        self.id_classifier = MLPClassification(input_size, hidden_size, num_classes)

    def forward(self, x):
        pr = self.pr_regressor(x)
        rt = self.rt_regressor(x)
        rr = self.rr_regressor(x)
        i = self.id_classifier(x)

        return pr, rt, rr, i

"""
