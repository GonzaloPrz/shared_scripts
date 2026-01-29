import pickle,json,os,sys,itertools
import pandas as pd
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import bootstrap, percentileofscore
from expected_cost.ec import CostMatrix
from confidenceinterval.bootstrap import bootstrap_ci
from scipy.stats import ttest_1samp, wilcoxon, shapiro

sys.path.append(str(Path(Path.home(),"scripts_generales"))) if "Users/gp" in str(Path.home()) else sys.path.append(str(Path(Path.home(),"gonza","scripts_generales")))

import utils

def _calculate_metric_diffs(indices, outputs1, y_dev1, outputs2, y_dev2, metrics, prob_type, cost_matrix):
    """
    Statistic function for bootstrap. Calculates differences for ALL metrics at once.
    """
    # Resample y, ensuring we don't operate on an empty or invalid slice
    resampled_y1 = y_dev1[..., indices].ravel()
    resampled_y2 = y_dev2[..., indices].ravel()

    # If a resample is degenerate (e.g., missing a class), metric calculation is impossible.
    # Return NaNs to signal this. The 'bca' method will fail, triggering our fallback.
    if np.unique(resampled_y1).shape[0] != np.unique(y_dev1).shape[0] or \
       np.unique(resampled_y2).shape[0] != np.unique(y_dev2).shape[0]:
        return np.full(len(metrics), np.nan)

    # Resample model outputs
    resampled_out1 = outputs1[..., indices, :].reshape(-1, outputs1.shape[-1])
    resampled_out2 = outputs2[..., indices, :].reshape(-1, outputs2.shape[-1])

    # Get metrics for both classifiers
    if prob_type == 'clf':
        metrics1, _ = utils.get_metrics_clf(resampled_out1, resampled_y1, metrics, cmatrix=cost_matrix)
        metrics2, _ = utils.get_metrics_clf(resampled_out2, resampled_y2, metrics, cmatrix=cost_matrix)
    else: # 'reg'
        metrics1 = utils.get_metrics_reg(resampled_out1, resampled_y1, metrics)
        metrics2 = utils.get_metrics_reg(resampled_out2, resampled_y2, metrics)

    # Return an array of differences
    return np.array([metrics1[m] - metrics2[m] for m in metrics])

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = config['feature_selection']
filter_outliers = config['filter_outliers']
n_boot = int(config["n_boot"])
problem_type = config["problem_type"]
calibrate = bool(config["calibrate"])
scoring_metrics = [config['scoring_metric']]
problem_type = config['problem_type']
version = config['version']

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
metrics_names = main_config["metrics_names"][problem_type]
data_file = main_config["data_file"][project_name]

try:
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
except:
    cmatrix = None

for scoring in scoring_metrics:
    best_models_file = f'best_best_models_{scoring}_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_calibrated_bayes.csv'.replace('__','_')
    if not hyp_opt:
        best_models_file = best_models_file.replace('_hyp_opt','')
    if not feature_selection:
        best_models_file = best_models_file.replace('_feature_selection','')
    if not calibrate:
        best_models_file = best_models_file.replace('_calibrated','')
    
    best_models = pd.read_csv(Path(results_dir,best_models_file))
    index_list = itertools.combinations(best_models.index,2)

    all_results = pd.DataFrame()
    
    for index in index_list:
        for y_label in y_labels:

            if best_models.loc[index[0],'task'] == best_models.loc[index[1],'task']:
                continue

            # Find best models for each task/dimension
            tasks = [best_models.loc[index[0],'task'], best_models.loc[index[1],'task']]
            dimensions = [best_models.loc[index[0],'dimension'], best_models.loc[index[1],'dimension']]
            best1 = best_models[(best_models.task == best_models.loc[index[0],'task']) & (best_models.dimension == best_models.loc[index[0],'dimension']) & (best_models.y_label == y_label)].iloc[0]
            best2 = best_models[(best_models.task == best_models.loc[index[1],'task']) & (best_models.dimension == best_models.loc[index[1],'dimension']) & (best_models.y_label == y_label)].iloc[0]
            
            if best1.empty or best2.empty:
                continue

            # Load data for both models
            IDs1, outputs1, y_dev1,_,_,_ = utils._load_data(results_dir,tasks[0],dimensions[0],y_label,best1.model_type,'',config, bayes=True, scoring=scoring)
            IDs2, outputs2, y_dev2,_,_,_ = utils._load_data(results_dir, tasks[1], dimensions[1], y_label, best2.model_type, '', config, bayes=True, scoring=scoring)

            # Align outputs based on IDs
            common_IDs = np.intersect1d(IDs1, IDs2)
            idx1 = np.isin(IDs1, common_IDs)
            idx2 = np.isin(IDs2, common_IDs)
            n = int(np.min((idx1.sum()/idx1.shape[0],idx2.sum()/idx2.shape[0])))
            outputs1 = outputs1[idx1].reshape((outputs1.shape[0],n,outputs1.shape[2]))
            y_dev1 = y_dev1[idx1].reshape((y_dev1.shape[0],n))
            outputs2 = outputs2[idx2].reshape((outputs2.shape[0],n,outputs2.shape[2]))
            y_dev2 = y_dev2[idx2].reshape((y_dev2.shape[0],n))

            # Ensure the datasets are comparable
            try:
                assert y_dev1.shape == y_dev2.shape, "y_dev shapes must match for paired comparison!"
            except:
                print(f"Shape mismatch for {tasks[0]}, {tasks[1]}")
                continue
            # Prepare data for bootstrap: a tuple of index arrays to resample
            data_indices = (np.arange(y_dev1.shape[-1]),)

            # Define the statistic function with data baked in
            stat_func = lambda indices: _calculate_metric_diffs(
                indices, outputs1, y_dev1, outputs2, y_dev2, 
                metrics_names, problem_type, cmatrix
            )

            # 1. Calculate the point estimate (the actual difference on the full dataset)
            point_estimates = stat_func(data_indices[0])

            # 2. Calculate the bootstrap confidence interval
            try:
                # Try the more accurate BCa method first
                res = bootstrap(
                    data_indices,
                    stat_func,
                    n_resamples=n_boot, # Use configured n_boot
                    method=config["bootstrap_method"],
                    vectorized=False,
                    random_state=42
                )
                bootstrap_method = config["bootstrap_method"]

            except ValueError as e:
                # If BCa fails (e.g., due to degenerate samples), fall back to percentile
                print(f"WARNING: {config['bootstrap_method']} method failed for {tasks}/{dimensions}/{y_label}. Falling back to 'percentile'. Error: {e}")
                res = bootstrap(
                    data_indices,
                    stat_func,
                    n_resamples=n_boot,
                    method='percentile',
                    vectorized=False,
                    random_state=42
                )
                bootstrap_method = 'percentile'
            # Store results for this comparison
            result_row = {
                "tasks": f'[{tasks[0]}, {tasks[1]}]',
                "dimensions": f'[{dimensions[0]}, {dimensions[1]}]',
                "y_label": y_label,
                "bootstrap method": bootstrap_method
            }
            
            for i, metric in enumerate(metrics_names):
                est = point_estimates[i]
                distribution = res.bootstrap_distribution[i] if est > 0 else -res.bootstrap_distribution[i]
                ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
                result_row[metric] = f"{est:.3f}, ({ci_low:.3f}, {ci_high:.3f})"
                result_row[f'p_value_{metric}'] = np.round(percentileofscore(distribution,0) / 100.0,3)
            
            if all_results.empty:
                all_results = pd.DataFrame(columns=result_row.keys())
            
            all_results.loc[len(all_results.index),:] = result_row

    # --- Save Final Results ---
    output_filename = Path(results_dir, f'ic_diff_{scoring}_bayes.csv') # Assumes one scoring metric for filename
    all_results.to_csv(output_filename, index=False)
    print(f"Confidence interval differences saved to {output_filename}")
    