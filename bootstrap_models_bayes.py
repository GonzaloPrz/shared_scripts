import pandas as pd
from pathlib import Path
from expected_cost.utils import *
import itertools
import json
import numpy as np
from scipy.stats import bootstrap

from expected_cost.ec import CostMatrix

import utils

##---------------------------------PARAMETERS---------------------------------##
config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = bool(config['n_iter'])
feature_selection = bool(config['feature_selection'])
n_boot = int(config["n_boot"])
calibrate = bool(config["calibrate"])
overwrite = bool(config["overwrite"])
filter_outliers = bool(config['filter_outliers'])
round_values = bool(config['round_values'])
cut_values = bool(config['cut_values'] > 0)
regress_out = len(config['covariates']) > 0
version = config['version']

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = config['y_labels']
test_size = config['test_size']
try:
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
except KeyError:
    cmatrix = None

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

scorings = [config["scoring_metric"]]

tasks = [folder.name for folder in Path(results_dir).iterdir() if folder.is_dir() and folder.name not in ['plots','rocs','feature_importance_bayes','feature_importance','final_models_bayes','final_models']]

output_filename = f'best_models_{scorings[0]}_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_filter_outliers_round_cut_shuffled_calibrated_bayes_{version}.csv'.replace('__','_')
                
if not hyp_opt:
        output_filename = output_filename.replace('_hyp_opt','')
if not feature_selection:
    output_filename = output_filename.replace('_feature_selection','')
if not filter_outliers:
    output_filename = output_filename.replace('_filter_outliers','')
if not round_values:
    output_filename = output_filename.replace('_round','')
if not cut_values:
    output_filename = output_filename.replace('_cut','')
if not shuffle_labels:
    output_filename = output_filename.replace('_shuffled','')
if not calibrate:
    output_filename = output_filename.replace('_calibrated','')
if (Path(results_dir,output_filename).exists()) & (not overwrite):
    all_results = pd.read_csv(Path(results_dir,output_filename))
else:
    all_results = pd.DataFrame()

for task in tasks:
    print(task)

    if not Path(results_dir,task).exists():
        continue

    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(dimension)
        y_labels_ = [folder.name for folder in Path(results_dir,task,dimension,kfold_folder).iterdir() if folder.is_dir()]
         
        for y_label in y_labels_:      
            #print(y_label)      
            path_ = Path(results_dir,task,dimension,kfold_folder,y_label,stat_folder)
            if not path_.exists():
                continue
            
            scorings += [folder.name for folder in path_.iterdir() if folder.is_dir()]

            for scoring in np.unique(scorings):
                #print(scoring)
                path = Path(path_,scoring,"hyp_opt" if hyp_opt else "","feature_selection" if feature_selection else "","filter_outliers" if filter_outliers else "","rounded" if round_values else "","cut" if cut_values else "","shuffle" if shuffle_labels else "")

                if not path.exists():
                    continue

                random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name] if config['test_size'] > 0 else []

                if len(random_seeds) == 0:
                    random_seeds = ['']

                for random_seed in random_seeds:
                    if random_seed == '':
                        random_seed = 'nan'

                    models = [filename.stem.split('_')[-1] for filename in Path(path,random_seed,version).glob('*.csv') if all(x not in filename.stem for x in ['train','test'])] 
                    
                    for model_type in models:
                        print(model_type)

                        if not overwrite and all_results.shape[0] > 0:
                            row = all_results[(all_results['task'] == task) & (all_results['dimension'] == dimension) & (all_results['model_type'] == model_type) & (all_results['y_label'] == y_label) & (all_results['random_seed_test'].astype(str) == str(random_seed))]
                            if len(row) > 0:
                                continue
                        
                        if not utils._build_path(results_dir,task,dimension,y_label,random_seed,f"outputs_{model_type}.npy",config,bayes=True,scoring=scoring).exists():
                            continue
                        
                        data_file = json.load(open(utils._build_path(results_dir,task,dimension,y_label,random_seed,"config.json",config,bayes=True,scoring=scoring),'rb'))['data_file']

                        _,outputs, y_dev,_,_,_ = utils._load_data(results_dir,task,dimension,y_label,model_type,random_seed,config,bayes=True,scoring=scoring)

                        problem_type = config['problem_type']
                        
                        metrics_names = main_config["metrics_names"][problem_type]
                        scoring_col = f'{scoring}_extremo'

                        extremo = 1 if any(x in scoring for x in ['error','norm']) else 0
                        ascending = any(x in scoring for x in ['error','norm'])

                        if (cmatrix is not None) or (np.unique(y_dev).shape[0] > 2):
                            metrics_names_ = list(set(metrics_names) - set(["roc_auc","f1","precision","recall"]))
                        else:
                            metrics_names_ = metrics_names
                            
                        # Prepare data for bootstrap: a tuple of index arrays to resample
                        data_indices = (np.arange(y_dev.shape[-1]),)

                        # Define the statistic function with data baked in
                        stat_func = lambda indices: utils._calculate_metrics(
                        indices, outputs, y_dev, 
                        metrics_names_, problem_type, cmatrix)

                        # 1. Calculate the point estimate (the actual difference on the full dataset)
                        try:
                            point_estimates = stat_func(data_indices[0])
                        except Exception as e:
                            print(f"ERROR calculating metrics for {task}/{dimension}/{y_label} with model {model_type}. Error: {e}")
                            continue
                            
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
                            "task": task,
                            "dimension": dimension,
                            "y_label": y_label,
                            "model_type": model_type,
                            "random_seed_test": random_seed if random_seed != '' else 'nan',
                            "bootstrap_method_dev": bootstrap_method,
                            "data_file": data_file
                        }
                        
                        for i, metric in enumerate(metrics_names_):
                            est = point_estimates[i]
                            ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
                            result_row[metric] = f"{est:.2f}, ({ci_low:.2f}, {ci_high:.2f})"
                        
                        if all_results.empty:
                            all_results = pd.DataFrame(result_row,index=[0])
                        else:
                            all_results.loc[all_results.shape[0],:] = result_row

                        all_results.to_csv(Path(results_dir,output_filename),index=False)

for scoring in np.unique(scorings):
    output_filename = f'best_models_{scoring}_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_filter_outliers_round_cut_shuffled_calibrated_bayes_{config["version"]}.csv'.replace('__','_')

    if not hyp_opt:
        output_filename = output_filename.replace('_hyp_opt','')
    if not feature_selection:
        output_filename = output_filename.replace('_feature_selection','')
    if not filter_outliers:
        output_filename = output_filename.replace('_filter_outliers','')
    if not round_values:
        output_filename = output_filename.replace('_round','')
    if not cut_values:
        output_filename = output_filename.replace('_cut','')
    if not shuffle_labels:
        output_filename = output_filename.replace('_shuffled','')
    if not calibrate:
        output_filename = output_filename.replace('_calibrated','')

    if not Path(results_dir,output_filename).exists():
        continue

    all_results = pd.read_csv(Path(results_dir,output_filename))

    all_results['random_seed_test'] = all_results['random_seed_test'].astype(str).apply(lambda x: str(x).lower())

    y_labels_ = all_results['y_label'].unique()
    best_best_models = pd.DataFrame(columns=all_results.columns)

    random_seeds_test = all_results['random_seed_test'].unique()

    scoring_col = f'{scoring}_extremo'
    extremo = 1 if any(x in scoring for x in ['error','norm']) else 0
    ascending = any(x in scoring for x in ['error','norm'])

    for random_seed_test in random_seeds_test: 
        for task in tasks:
            if not Path(results_dir,task).exists():
                continue
            
            dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
            for dimension in dimensions:
                
                y_labels_ = all_results['y_label'].unique()
                
                for y_label in y_labels_:
                    best_best_models_ = all_results[(all_results['task'] == task) & (all_results['y_label'] == y_label) & (all_results['dimension'] == dimension) & (all_results['random_seed_test'].astype(str) == str(random_seed_test))]

                    #if best_best_models_.shape[0] == 0:
                    #    continue
                    try:
                        best_best_models_[scoring_col] = best_best_models_[scoring].apply(lambda x: float(x.split(', ')[0]))
                    except:
                        best_best_models_[scoring_col] = best_best_models_[f'{scoring}_score'].apply(lambda x: float(x.split(', ')[0]))

                    best_best_models_.dropna(subset=[scoring_col], inplace=True)
                    
                    try:
                        best_best_models_append = best_best_models_.sort_values(by=scoring_col,ascending=ascending).iloc[0]
                    except:
                        print(f"WARNING: No valid models found for {task}/{dimension}/{y_label} with random seed {random_seed_test}. Skipping...")
                        continue

                    best_best_models.loc[best_best_models.shape[0],:] = best_best_models_append

    try:
        best_best_models[scoring_col] = best_best_models[scoring].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
    except:
        best_best_models[scoring_col] = best_best_models[f'{scoring}_score'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))

    best_best_models = best_best_models.sort_values(by=['y_label',scoring_col],ascending=ascending).reset_index(drop=True)

    y_labels_ = all_results['y_label'].unique()
    random_seeds_test = all_results['random_seed_test'].unique()
    best_best_best_models = pd.DataFrame(columns=best_best_models.columns)
    
    for y_label,task,random_seed_test in itertools.product(y_labels_,tasks,random_seeds_test):
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            try:
                idx = best_best_models[(best_best_models['y_label'] == y_label) & (best_best_models['task'] == task) & (best_best_models['dimension'] == dimension) & (best_best_models['random_seed_test'].astype(str) == str(random_seed_test))].index[0]
                best_best_best_models.loc[best_best_best_models.shape[0],:] = best_best_models.loc[idx,:]
            except:
                continue

    best_best_best_models.to_csv(Path(results_dir,f'best_{output_filename}'),index=False)