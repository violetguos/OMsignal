# OMSignal - Block 1

This is the repository containing the code for Block 1 of the OMSignal project from team `b1pomt5`:

* Saber Benchalel
* Arlie Coles
* Rémi Lussier St-Laurent

Our approach uses four models: one to accomplish each of the `pr`, `rt`, `rr`, and `rt`tasks. 

## Directory structure

`data/` contains dummy data that we generated for our own local development. By default, when running anything on the cluster, this data is ignored and the real data is used instead.

`evaluation/` contains scripts needed for the official TA evaluation of our models.

`figures/` contains several example figures from our analysis, and is the default directory where any newly generated figures are put.

`models/` contains several trained models and is the default directory where new saved models are put.

`logs/` contains logs from some experiments (logs from hyperparameter tuning are not included, since the number of generated logs was enormous and grew exponentially), and is the default directory where any newly generated logs are put. By default, every instance of training will generate a log.

`omsignal/` is the main code directory, which contains all code needed for model training, experiments, utilities, and visualization.

## Configuration

It is sometimes useful to be able to access the above directories from anywhere in the project. To help with this, we also include a `config.py` file, which can be imported by code in the `omsignal/` package and which includes variables to some important paths in the file structure:

* `ROOT_DIR` - a path to the root of the repository;
* `DATA_DIR` - a path to `data/`;
* `LOG_DIR` - a path to `log/`;
* `MODELS_DIR` - a path to `/models/`;
* `FIGURES_DIR` - a path to `figures/`;
* `CLUSTER_DIR` - a path to the real data directory on the Helios cluster

## Running on the cluster

### Scheduling a job

Several `.pbs` scripts are intended to be put into the Helios queue, wrapping around a `.py` script. That can be done like this:

```
msub [script_name].pbs
```

This will start the job from inside a Singularity container and route the `stdout` and `stderr` to text files in the same directory as the script.

### Running from an interactive container

Other times, for short jobs, it is easier to simply launch a debugging session on the cluster and run the `.py` script directly (if the job will take less than 15 minutes). This method will put the user automatically in priority, if a GPU is available. It can be done like this:

```
mdebug
s_exec
python [script_name].py
```

This opens a "debugging" session, opens an interactive container shell, and starts the script. This time, `stdout` and `stderr` are displayed as normal.
