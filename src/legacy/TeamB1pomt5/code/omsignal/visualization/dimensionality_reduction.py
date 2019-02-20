'''
Functions for visualization of data
via dimensionality reduction.
'''
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join('..', '..'))   # Hack - fix package management later

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.legacy.TeamB1pomt5.code.omsignal.utils.preprocessor import Preprocessor
from src.legacy.TeamB1pomt5.code.omsignal.utils.dataloader_utils import import_OM

def plot_scatter(x, y, c, title, output_path, xerr=None, yerr=None):
    plt.title(title)
    plt.scatter(x, y, c=c, zorder=1)
    if xerr is not None:
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, capsize=2,\
            elinewidth=0.5, ls='none', zorder=-1)
    plt.savefig(output_path)

def get_means_stdevs(x, y, ids):
    # Collect points belonging to each participant
    points = {p:[] for p in set(ids)}
    for i, x_i in enumerate(x):
        y_i = y[i]
        points[ids[i]].append(np.array([x_i, y_i]))
    # Average the points together/collect stdevs
    centers, stdevs = {}, {}
    for k, v in points.items():
        v = np.array(v)
        centers[k] = np.mean(v, axis=0)
        stdevs[k] = np.std(v, axis=0)
    ids = list(centers.keys())
    x = np.array(list(centers.values()))[:,0]
    y = np.array(list(centers.values()))[:,1]
    xerr = np.array(list(stdevs.values()))[:,0]
    yerr = np.array(list(stdevs.values()))[:,1]
    return x, y, xerr, yerr, ids

def plot_pca(data, ids, output_path, centroid=False):
    pca = PCA(n_components=2)   # Reduce to 2 dimensions
    reduced = pca.fit_transform(data)
    x = reduced[:,0]
    y = reduced[:,1]
    xerr, yerr = None, None
    if centroid:
        x, y, xerr, yerr, ids = get_means_stdevs(x, y, ids)
    plot_scatter(x, y, ids, 'PCA visualization', output_path, xerr=xerr, yerr=yerr)

def plot_tsne(data, ids, output_path, centroid=False):
    reduced = TSNE(n_components=2).fit_transform(data)
    x = reduced[:,0]
    y = reduced[:,1]
    xerr, yerr = None, None
    if centroid:
        x, y, xerr, yerr, ids = get_means_stdevs(x, y, ids)
    plot_scatter(x, y, ids, 'T-SNE visualization', output_path, xerr=xerr, yerr=yerr)

def plot_pca_tsne(data, ids, output_path, centroid=False):
    pca = PCA(n_components=100)   # Reduce to 50 dimensions before running t-SNE
    pca_reduced = pca.fit_transform(data)
    tsne_reduced = TSNE(n_components=2).fit_transform(pca_reduced)
    x = tsne_reduced[:,0]
    y = tsne_reduced[:,1]
    xerr, yerr = None, None
    if centroid:
        x, y, xerr, yerr, ids = get_means_stdevs(x, y, ids)
    plot_scatter(x, y, ids, 'T-SNE after PCA visualization', output_path, xerr=xerr, yerr=yerr)

if __name__ == '__main__':
    dr_methods = ['pca', 'tsne', 'pca_tsne']

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Generate line graph visualizations of the data.')
    parser.add_argument('dr_method', help='Dimensionality reduction method. \
        Choices are: {}.'.format(dr_methods))
    parser.add_argument('--cluster', help='Flag for running on the Helios cluster. \
        If flagged, will use real data; otherwise, will use dummy data.', action='store_true')
    parser.add_argument('--centroid', help='Plot centroids of each participant instead \
        of every point.', action='store_true')
    args = parser.parse_args()

    # Check arguments
    if args.dr_method not in dr_methods:
        raise ValueError('Invalid dimensionality reduction method. \
            Choices are: {}.'.format(dr_methods))

    # Load real vs dummy data if we are on cluster vs local, respectively
    ecg, _, _, _, ids = import_OM('train', cluster=args.cluster)

    # Preprocess ecg data before plotting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocessor()
    preprocess.to(device)
    ecg_normalized = preprocess(torch.from_numpy(ecg).reshape(\
        ecg.shape[0],-1,ecg.shape[1])).flatten(start_dim=1).numpy()

    # Reduce dimension and plot
    if args.centroid:
        fig_name = 'train_{}_centroids.png'.format(args.dr_method)
    else:
        fig_name = 'train_{}.png'.format(args.dr_method)
    output_path = os.path.join('..', '..', 'figures', fig_name)
    
    if args.dr_method == 'pca':
        plot_pca(ecg_normalized, ids, output_path, centroid=args.centroid)
    elif args.dr_method == 'pca_tsne':
        plot_pca_tsne(ecg_normalized, ids, output_path, centroid=args.centroid)
    else:
        plot_tsne(ecg_normalized, ids, output_path, centroid=args.centroid)
    