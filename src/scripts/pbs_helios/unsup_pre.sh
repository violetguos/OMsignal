#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/block2/src"
export PYTHONPATH="${PYTHONPATH}:src"

# To run this locally for debugging python syntax errors
python scripts/unsupervised_pretraining.py ../algorithm/autoencoder_input.in model_input.in