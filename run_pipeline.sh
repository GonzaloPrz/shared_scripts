#!/bin/bash

# Run Python scripts with all passed arguments
python "/Users/gp/scripts/shared/train_models.py" "$@"
python "/Users/gp/scripts/shared/bootstrap_models_bca.py"
python "/Users/gp/scripts/shared/test_models.py"
python "/Users/gp/scripts/shared/report_best_models.py"
python "/Users/gp/scripts/shared/train_final_model.py"
python "/Users/gp/scripts/shared/plot_scores.py"
python "/Users/gp/scripts/shared/compare_dev_test_models.py"

# Display used parameters for logging
echo "Pipeline executed with arguments: $@"
