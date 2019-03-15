#!/bin/bash

cd "${PBS_O_WORKDIR}"
source /rap/jvb-000-aa/COURS2019/etudiants/common.env

# PROJECT_PATH will be changed to the master branch of your repo
PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt1/code'

RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2pomt1/code/evaluation/'
DATA_FILE='/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/MILA_ValidationLabeledData.dat'

s_exec python $PROJECT_PATH/evaluation/eval.py --dataset=$DATA_FILE --results_dir=$RESULTS_DIR