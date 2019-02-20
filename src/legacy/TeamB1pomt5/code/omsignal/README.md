# Code directory

This directory is the main code directory, containing everything needed to run experiments, train networks, and visualize data, as well as several utilities.

## Directory structure

`experiments/` contains code needed to perform hyperparameter gridsearch, to train a network from scratch given certain hyperparameters, and to score the performance of all four trained models.

`utils/` contains several utility functions for network training and other data transformation.

`visualization/` contains several functions for visualization of data, including learning curve and dimensionality reduction plots.

## Network files

This folder also contains the main classes needed for the construction of our neural networks.

`base_networks.py` contains `CNNRegressor` and `CNNClassifier` classes, which can be used as-is (as we have), or imported into more complicated networks as submodules. We have also included several other "basic" networks that we did not end up using, but that future teams may find helpful.

`om_networks.py` contains the `CNNRank` class, which is a CNN designed to predict the respective rank of pairwise samples (as described in our report as used for `RT-Ranker`). We have also included several more complex multitask models that we did not end up using, but that future teams might find helpful.
