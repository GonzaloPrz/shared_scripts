import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import itertools
from joblib import Parallel, delayed
import sys,os,json
from scipy.stats import bootstrap
import tqdm

from pingouin import compute_bootci 

from expected_cost.ec import CostMatrix

import utils

##---------------------------------PARAMETERS---------------------------------##
config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]

scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = config['n_iter'] > 0
feature_selection = config['n_iter_features'] > 0
filter_outliers = config['filter_outliers']
n_models = float(config["n_models"])
n_boot = int(config["n_boot"])
calibrate = bool(config["calibrate"])
overwrite = bool(config["overwrite"])
parallel = bool(config["parallel"])
bootstrap_method = config["bootstrap_method"]
regress_out = config["regress_out"]

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = config['scoring_metrics']
if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]

problem_type = config['problem_type']
metrics_names_ = main_config["metrics_names"][problem_type]

if problem_type == 'clf':
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
else:
    cmatrix = None
##---------------------------------PARAMETERS---------------------------------##
for task,y_label,scoring in itertools.product(tasks,y_labels,scoring_metrics):    

    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    #dimensions = ["verbosity__word_properties__talking_intervals"]
    for dimension in dimensions:
        path = Path(results_dir,task,dimension, kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '',"regress_out" if regress_out else "","shuffle" if shuffle_labels else "")
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        random_seeds.append('') # Add empty string to the list of random seeds
        
        for random_seed in random_seeds:
            models = [f.name.split('_')[2].replace('.csv','') for f in Path(path,random_seed).glob('all_models_*.csv') if 'dev' not in f.name]
            if len(models) == 0:
                continue

            for model in models:
                all_results = pd.DataFrame()

                filename_to_save = f'all_models_{model}_dev_{config["bootstrap_method"]}_calibrated.csv'
                if not calibrate:
                    filename_to_save = filename_to_save.replace('_calibrated','')
                if config['n_models'] != 0:
                    filename_to_save = filename_to_save.replace('all_models','best_models').replace('.csv',f'_{scoring}.csv')

                if Path(path,random_seed,filename_to_save).exists() and overwrite == False:
                    print(f"Bootstrapping already done for {task} - {y_label} - {model} - {dimension}. Skipping...")
                    continue
                
                if not Path(path,random_seed,f'all_models_{model}.csv').exists():
                    continue
                
                print(task,model,dimension,y_label)

                all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))

                outputs = pickle.load(open(Path(path,random_seed,f'cal_outputs_{model}.pkl' if calibrate else f'outputs_{model}.pkl'),'rb'),fix_imports=False, encoding="latin1") 
                
                y_dev = pickle.load(open(Path(path,random_seed,'y_dev.pkl'),'rb'),fix_imports=False, encoding="latin1")
                
                if np.unique(y_dev).shape[0] > 2:
                    metrics_names = list(set(metrics_names_) - set(['roc_auc','presicion','recall','f1']))
                else:
                    metrics_names = metrics_names_

                IDs_dev = pickle.load(open(Path(path,random_seed,'IDs_dev.pkl'),'rb'),fix_imports=False, encoding="latin1")

                scorings = np.empty(outputs.shape[1])
                
                if config['n_models'] == 0:
                    n_models = outputs.shape[1]
                    all_models_bool = True
                else:
                    all_models_bool = False
                    if config['n_models'] < 1:
                        n_models = int(outputs.shape[1]*n_models)

                    for i in range(outputs.shape[1]):
                        scorings_i = np.empty((outputs.shape[0],outputs.shape[2]))
                        for j,r in itertools.product(range(outputs.shape[0]),range(outputs.shape[2])):
                            if problem_type == 'clf':
                                metrics, _ = utils.get_metrics_clf(outputs[j,i,r], y_dev[j,r], [scoring], cmatrix)
                                scorings_i[j,r] = metrics[scoring]
                            else:
                                metrics = utils.get_metrics_reg(outputs[j,i,r], y_dev[j,r],[scoring])
                                scorings_i[j,r] = metrics[scoring]
                        scorings[i] = np.nanmean(scorings_i.flatten())
                    
                    scorings = scorings if any(x in scoring for x in ['norm','error']) else -scorings

                    best_models = np.argsort(scorings)[:n_models]
                
                    all_models = all_models.iloc[best_models].reset_index(drop=True)
                    all_models['idx'] = best_models
                    outputs = outputs[:,best_models]
                    n_models = len(best_models)

                def parallel_process(index):
                    # Prepare data for bootstrap: a tuple of index arrays to resample
                    data_indices = (np.arange(y_dev.shape[-1]),)

                    # Define the statistic function with data baked in
                    stat_func = lambda indices: utils._calculate_metrics(
                        indices, outputs[:,index], y_dev, 
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
                        print(f"WARNING: BCa method failed for {tasks}/{dimensions}/{y_label}. Falling back to 'percentile'. Error: {e}")
                        res = bootstrap(
                            data_indices,
                            stat_func,
                            n_resamples=n_boot,
                            method='percentile',
                            vectorized=False,
                            random_state=42
                        )
                        bootstrap_method = 'percentile'

                    result_row = dict(all_models.iloc[index])
                    result_row.update({'bootstrap_method':bootstrap_method})
                    for i, metric in enumerate(metrics_names):
                        est = point_estimates[i]
                        ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
                        result_row.update({metric: f"{est:.5f}, ({ci_low:.5f}, {ci_high:.5f})"})
                    
                    return result_row
                
                parallel_results = Parallel(n_jobs=1 if parallel else 1)(delayed(parallel_process)(index) for index in np.arange(n_models))
                all_results = pd.concat((pd.DataFrame(parallel_result,index=[0]) for parallel_result in parallel_results),ignore_index=True)
                
                all_results.to_csv(Path(path,random_seed,filename_to_save))