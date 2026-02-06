import pickle,os,json,sys,itertools
from pathlib import Path
import pandas as pd
import numpy as np

from expected_cost.calibration import *
from expected_cost.psrcal_wrappers import LogLoss
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal

sys.path.append(str(Path(Path.home(),"scripts_generales"))) if "Users/gp" in str(Path.home()) else sys.path.append(str(Path(Path.home(),"gonza","scripts_generales")))

def flatten_except(arr, axes_to_keep):
    # Ensure axes_to_keep is a tuple
    if isinstance(axes_to_keep, int):
        axes_to_keep = (axes_to_keep,)
    
    # Move the axes to keep to the front
    new_axes_order = list(axes_to_keep) + [i for i in range(arr.ndim) if i not in axes_to_keep]
    arr = np.moveaxis(arr, new_axes_order, range(arr.ndim))
    
    # Get the new shape: keep the selected axes, flatten the rest
    shape_kept = [arr.shape[i] for i in range(len(axes_to_keep))]
    flattened_size = -1  # Let NumPy infer the flattened dimension size
    
    return arr.reshape(*shape_kept, flattened_size)

import utils

metric = LogLoss 
calmethod = AffineCalLogLoss
deploy_priors = None
calparams = {'bias': True, 'priors': deploy_priors}

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
problem_type = config["problem_type"]

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

scoring_metrics = [main_config['scoring_metrics'][project_name]]
problem_type = main_config['problem_type'][project_name]

for scoring in scoring_metrics:

    extremo = "sup" if "norm" in scoring else "inf"
    ascending = True if extremo == "sup" else False

    best_models_file = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_shuffled.csv'.replace('__','_')
    if not hyp_opt:
        best_models_file = best_models_file.replace('_hyp_opt','')
    if not feature_selection:
        best_models_file = best_models_file.replace('_feature_selection','')
    if not shuffle_labels:
        best_models_file = best_models_file.replace('_shuffled','')

    best_models = pd.read_csv(Path(results_dir,best_models_file))

    for r, row in best_models.iterrows():
        print(row["task"], row["dimension"])
        path_to_results = Path(results_dir, row.task, row.dimension, scaler_name, kfold_folder, row.y_label,stat_folder, "hyp_opt" if hyp_opt else "", "feature_selection" if feature_selection else "", 'filter_outliers' if filter_outliers and problem_type == 'reg' else '')

        model_name = row.model_type
        if row.random_seed_test == np.nan:
            random_seed = ""
        else:
            random_seed = row.random_seed_test

        try:
            model_index = pd.read_csv(Path(path_to_results, random_seed, f"all_models_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending).index[0]
        except:
            model_index = pd.read_csv(Path(path_to_results, random_seed, f"best_models_{scoring}_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending).index[0]

        outputs = pickle.load(open(Path(path_to_results, random_seed, f"outputs_{model_name}.pkl"), "rb"))[:,model_index]

        cal_outputs = np.zeros_like(outputs) 

        y_true = pickle.load(open(Path(path_to_results,random_seed, f"y_dev.pkl"), "rb")).astype(int)

        for j,r in itertools.product(range(outputs.shape[0]),range(outputs.shape[1])):
            cal_outputs[j,r] = calibration_train_on_test(outputs[j,r], y_true[j,r], calmethod, calparams)
        
        overall_perf = metric(outputs.reshape(outputs.shape[0]*outputs.shape[1]*outputs.shape[2],outputs.shape[3]), y_true.flatten(), priors=deploy_priors, norm=True)
        overall_perf_after_cal = metric(cal_outputs.reshape(cal_outputs.shape[0]*cal_outputs.shape[1]*cal_outputs.shape[2],cal_outputs.shape[3]), y_true.flatten(), priors=deploy_priors, norm=True)
        cal_loss = overall_perf-overall_perf_after_cal
        rel_cal_loss = 100*cal_loss/overall_perf

        print(f"Overall performance before calibration: {overall_perf:4.4f} for {model_name} in {row.task} - {row.dimension} - {row.y_label}")
        print(f"Overall performance after calibration: {overall_perf_after_cal:4.4f} for {model_name} in {row.task} - {row.dimension} - {row.y_label}")
        print(f"Calibration loss: {cal_loss:4.4f} for {model_name} in {row.task} - {row.dimension} - {row.y_label}")
        print(f"Relative calibration loss: {rel_cal_loss:4.2f} % for {model_name} in {row.task} - {row.dimension} - {row.y_label}") 