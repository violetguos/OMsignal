# Utilities

This directory contains utility functions used in various parts of our analysis. 

## Data transformation

### Augmentation

`augmentation.py` contains PyTorch-compatible transformations to be used for data augmentation. It includes:

* `CircShift` - shifts the ECG signal by a specified amount.
* `Negate` - negates the value of the signal.
* `ReplaceNoise` - replaces a sub-window of the signal with controlled noise matching the mean and variance of the signal.
* `DropoutBurst` - zeros out the signal in a 48ms window around a time instant, simulating periods of weak signal (as suggested in [this paper](https://arxiv.org/pdf/1710.06122.pdf)).


There are also randomized versions of these transformations:

* `RandomCircShift` - randomly shifts the ECG signal between -125 and 125 positions (+/- 1 second).
* `RandomNegate` - randomly negates the value of the signal.
* `RandomDropoutBurst` - zeros out the signal in a random 48ms window.

### Preprocessing

`preprocessor.py` is a TA-provided class that performs normalizing and baseline-stabilizing preprocessing.

`fft_utils.py` is a TA-provided set of functions for applying the fast Fourier transform (FFT) on an ECG signal.

`spectrogram_utils.py` is a TA-provided set of functions for transforming an ECG signal into a spectrogram.

## Data loading and i/o

In order to be able to develop locally, we generate dummy data by perturbing the values of the original data. The script to generate this dummy data is `dummy_data.py`, wrapped with the Moab script to be run on the cluster, `generate_dummy_data.pbs`.

`memfile_utils.py` is a TA-provided set of functions for reading and writing numpy `memmap`files, which is the format of the original data.

`dataloader_utils.py` contains functions for the construction of `Dataset` classes. It includes `Dataset`s for ranking and non-ranking tasks, and other utilities for loading and cleaning data from disk.

## Training

`pytorch_utils.py` contains functions needed for neural network training, for the `PR`, `RT`, and `ID` tasks. 

`rr_stdev.py` contains functions needed for regression training, for the `RR` task.