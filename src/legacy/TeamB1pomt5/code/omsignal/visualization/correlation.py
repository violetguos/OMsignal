'''
This file tests and produces plot to test the adequacy of two hypothesis: 
1) The regression targets are normally distributed
2) The rank of our regression targets are correlated
'''
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from omsignal.utils.preprocessor import Preprocessor
from omsignal.utils.dataloader_utils import multitask_train_valid
import scipy.stats as stats
import matplotlib.pyplot as plt
from  sklearn import preprocessing

if __name__ == '__main__':

    cluster = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, X_valid, y_train, y_valid = multitask_train_valid(cluster = cluster)
    #y is pr_mean, rt_mean, rr_stdDev, userId

    preprocess = Preprocessor()
    preprocess.to(device)
    X_train = preprocess(torch.from_numpy(X_train)).numpy()
    X_valid = preprocess(torch.from_numpy(X_valid)).numpy()

    PR_mean_normalized = preprocessing.normalize(y_train[:,0].reshape((1,-1)))
    RT_mean_normalized = preprocessing.normalize(y_train[:,1].reshape((1,-1)))
    RR_StdDev_normalized = preprocessing.normalize(y_train[:,2].reshape((1,-1)))    

    fig, axes = plt.subplots(nrows= 2, ncols =2)
    fig.delaxes(axes[1,1])
    plt.subplots_adjust( hspace=0.75, wspace = 0.75)

    stats.probplot(PR_mean_normalized.ravel(), dist = "norm", plot=axes[0,0])
    axes[0,0].set_title("PR_Mean QQ-plot")

    stats.probplot(RT_mean_normalized.ravel(), dist = "norm", plot=axes[0,1])
    axes[0,1].set_title("RT_Mean QQ-plot")

    stats.probplot(RR_StdDev_normalized.ravel(), dist ="norm",plot=axes[1,0])
    axes[1,0].set_title("RR_StdDev QQ-plot")

    plt.savefig(os.path.join('..', '..', 'figures', "QQ_plots.png"))
    
    fig , axes = plt.subplots(2,2)
    fig.delaxes(axes[1,1])
    plt.subplots_adjust( hspace=0.75, wspace = 0.75)

    PR_rank = stats.rankdata(PR_mean_normalized)/y_train.shape[0]
    RT_rank = stats.rankdata(RT_mean_normalized)/y_train.shape[0]
    RR_rank = stats.rankdata(RR_StdDev_normalized)/y_train.shape[0]

    axes[0,0].scatter(x=PR_rank, y=RT_rank)
    axes[0,0].set_title("PR Quantile vs RT Quantile")
    axes[0,0].set_xlabel("PR Quantile")
    axes[0,0].set_ylabel("RT Quantile")

    axes[0,1].scatter(x=PR_rank, y=RR_rank)
    axes[0,1].set_title("PR Quantile vs RR Quantile")
    axes[0,1].set_xlabel("PR Quantile")
    axes[0,1].set_ylabel("RR Quantile")

    axes[1,0].scatter(x=RT_rank, y=RR_rank)
    axes[1,0].set_title("RT Quantile vs RR Quantile")
    axes[1,0].set_xlabel("RT Quantile")
    axes[1,0].set_ylabel("RR Quantile")

    plt.savefig(os.path.join('..', '..', 'figures', "Rank correlations.png")) 
