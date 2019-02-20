# Experiments

This directory contains scripts for hyperparameter search for our three neural networks (`PR_CNN`, `RT_Ranker`, and `ID_CNN`), and a script to get the overall score across all four prediction tasks.

## Hyperparameter search

`pr_gridsearch.pbs`, `rt_gridsearch.pbs`, and `id_gridsearch.pbs` are Moab scripts to be run on the Helios cluster that call `pr_gridsearch.py`, `rt_gridsearch.py`, and `id_gridsearch.py` respectively. They can simply be run this way, for example:

```
msub pr_gridsearch.pbs
```

Each of these scripts searches over a grid of possible hyperparameters and saves the model that produced the lowest validation error into the `models/` directory at the root of the repository, as `best_PR_CNN.pt`, `best_RT_Ranker.pt`, and `best_ID_CNN.pt` respectively. Additionally, a log of the best hyperparameter values is saved into the `logs/` directory as `PR_best_hyperparams.txt`, `RT_best_hyperparams.txt`, and `ID_best_hyperparams.txt` respectively.

## Scoring

`show_score.py` is a script that loads the three best neural network models as found by hyperparameter search and the best model for the RR task (which has been handled separately), and returns the overall score, the Kendall-Tau regression scores for the PR, RT, and RR tasks, and the macro-average recall for the ID task. It takes no arguments and can be run like this:

```
python show_score.py
```

## Training from scratch

`train_task.py` is a script designed to train a neural network from scratch, according to hard-coded hyperparameters in the file. It can be run like this:

```
python train_task.py [task_name] --combine
```

where the arguments are:

* `task_name` - the task for which a network needs to be trained. Can be one of `PR`, `RT`, or `ID`.
* `--combine` - a switch that, if supplied, will train a network on the combined training and validation data. This is useful for training models to be used for the final evaluation against the unseen test set.
