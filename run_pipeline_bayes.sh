#!/bin/bash

# Activate virtual environment
source "/Users/gp/miniconda3/envs/ml-env/bin/activate"

# Run Python scripts with all passed arguments
python "/Users/gp/shared_scripts/train_models/train_models_bayes.py" "$@"
python "/Users/gp/shared_scripts/eval_models/bootstrap_models_bayes.py"
python "/Users/gp/shared_scripts/train_models/train_final_model_bayes.py"
python "/Users/gp/shared_scripts/eval_models/independent_test_set.py"
python "/Users/gp/shared_scripts/eval_models/test_final_model_bayes.py"