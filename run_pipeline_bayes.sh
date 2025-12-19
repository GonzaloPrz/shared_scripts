#!/bin/bash

# Activate virtual environment
source "/Users/gp/gonza-env/Scripts/activate"

# Run Python scripts with all passed arguments
python "/Users/gp/scripts/shared/train_models_bayes.py" "$@"
python "/Users/gp/scripts/shared/bootstrap_models_bayes.py"
python "/Users/gp/scripts/shared/train_final_model_bayes.py"
python "/Users/gp/scripts/shared/test_final_model_bayes.py"
