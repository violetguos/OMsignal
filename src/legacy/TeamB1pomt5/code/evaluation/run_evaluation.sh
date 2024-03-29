#!/bin/bash

cd "${PBS_O_WORKDIR}"
source /rap/jvb-000-aa/COURS2019/etudiants/common.env

# PROJECT_PATH will be changed to the master branch of your repo
PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/omsignal/'

# RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/projects/omsignal/evaluation'
RESULTS_DIR = ''
DATA_FILE='/rap/jvb-000-aa/COURS2019/etudiants/data/omsignal/myHeartProject/sample_test.dat'

s_exec python $PROJECT_PATH/evaluation/eval.py --dataset=$DATA_FILE --results_dir=$RESULTS_DIR