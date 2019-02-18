#!/bin/bash

# The objective of this script is to load the python program that is inside of the singularity container
# The line below removes all previously loaded software modules from the current environment
module --force purge
export PATH=$PATH:/opt/software/singularity-3.0/bin/
singularity exec $SINGULARITY_ARGS /miniconda/bin/python "$@"