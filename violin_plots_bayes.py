import seaborn as sns
import pandas as pd
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings, json, os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import bootstrap

from expected_cost.ec import CostMatrix

import utils

warnings.filterwarnings('ignore')

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config['project_name']
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = config['n_iter'] > 0
feature_selection = bool(config['feature_selection'])
filter_outliers = config['filter_outliers']
test_size = float(config['test_size'])
n_boot = int(config['n_boot'])

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path('D:/CNC_Audio/gonza/results', project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

scoring = config['scoring_metric']
problem_type = config['problem_type']
metrics_names = main_config['metrics_names'][problem_type]

cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

# Set the style for the plots

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

data_to_plot = pd.DataFrame()
best_models_filename = f'best_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_bayes.csv'.replace('__','_')
best_models = pd.read_csv(Path(results_dir,best_models_filename))

tasks = best_models['task'].unique()
dimensions = best_models['dimension'].unique()
y_label = best_models['y_label'].unique()
random_seeds_test = best_models['random_seed_test'].unique()
for task,dimension,y_label,random_seed_test in itertools.product(tasks,dimensions,y_label,random_seeds_test):
    if best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label) & (best_models['random_seed_test'].astype(str) == str(random_seed_test))].shape[0] == 0:
        continue

    best_models_ = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label) & (best_models['random_seed_test'].astype(str) == str(random_seed_test))]

    model_name = best_models_.iloc[0]['model_type']

    data_append = pd.DataFrame()
    Path(results_dir,'plots','bayes').mkdir(parents=True, exist_ok=True)

    print(task, dimension)
    path_to_results = Path(results_dir, task, dimension, scaler_name, kfold_folder, y_label, stat_folder,'bayes',scoring,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')

    if str(random_seed_test) == 'nan':
        random_seed = ''
    else:
        random_seed = random_seed_test
                        
    outputs_filename = f'outputs_{model_name}.pkl'
    outputs_ = pickle.load(open(Path(path_to_results, random_seed, outputs_filename), 'rb'))
    y_true_ = pickle.load(open(Path(path_to_results,random_seed, f'y_dev.pkl'), 'rb'))
    
    data_indices = (np.arange(y_true_.shape[-1]),)

    stat_func = lambda indices: utils._calculate_metrics(
                indices, outputs_, y_true_, 
                metrics_names, problem_type, cmatrix
            )
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

    metrics_dict = dict((metric,np.empty((outputs_.shape[0],outputs_.shape[1],n_boot))) for metric in metrics_names)
    
    #Check whether IDs and IDs_shuffle are the same

    for i,metric in enumerate(metrics_names):
        
        metrics_dict[metric] = res.bootstrap_distribution[i]

        data_append[metric] = metrics_dict[metric]
        data_append['dimension'] = dimension
        data_append['task'] = task

    if data_to_plot.empty:
        data_to_plot = data_append
    else:
        data_to_plot = pd.concat((data_to_plot,data_append),axis=0)
#        data_to_plot.to_csv(Path(results_dir,filename_to_save), index=False)
data_to_plot = data_to_plot[((data_to_plot["task"] == "Animales") & (data_to_plot["dimension"] == "properties")) |
                            ((data_to_plot["task"] == "nps") & (data_to_plot["dimension"] == "mmse")) |
                            ((data_to_plot["task"] == "nps") & (data_to_plot["dimension"] == "executive")) |
                            ((data_to_plot["task"] == "brain") & (data_to_plot["dimension"] == "norm_brain_lit")) |
                            ((data_to_plot["task"] == "connectivity") & (data_to_plot["dimension"] == "networks")) 
                            ]

data_to_plot["dimension"] = data_to_plot["dimension"].map({"properties":"Fluency\n classifier","mmse": "Screening\n classifier","executive":"Executive\n classifier","norm_brain_lit":"Structural\n classifier", "networks":"Functional\n classifier"})
for metric in metrics_names:
    # Get mean values
    mean_animales = float(
        best_models[
            (best_models['task'] == 'Animales') &
            (best_models['dimension'] == 'properties')
        ][metric].values[0].split(', (')[0]
    )

    mean_mmse = float(
        best_models[
            (best_models['task'] == 'nps') &
            (best_models['dimension'] == 'mmse')
        ][metric].values[0].split(', (')[0]
    )

    mean_executive = float(
        best_models[
            (best_models['task'] == 'nps') &
            (best_models['dimension'] == 'executive')
        ][metric].values[0].split(', (')[0]
    )

    mean_brain = float(
        best_models[
            (best_models['task'] == 'brain') &
            (best_models['dimension'] == 'norm_brain_lit')
        ][metric].values[0].split(', (')[0]
    )
    
    mean_conn = float(
        best_models[
            (best_models['task'] == 'connectivity') &
            (best_models['dimension'] == 'networks')
        ][metric].values[0].split(', (')[0]
    )
    means = np.array([mean_animales, mean_mmse, mean_executive, mean_brain, mean_conn])
    
    # Get CI lower and upper bounds
    def extract_ci(task, dimension):
        val_str = best_models[
            (best_models['task'] == task) &
            (best_models['dimension'] == dimension)
        ][metric].values[0]
        # Split: e.g. "0.80, (0.75, 0.84)"
        ci_part = val_str.split(', (')[1].replace(')', '')
        ci_low, ci_high = ci_part.split(', ')
        return float(ci_low), float(ci_high)

    ci_animales = extract_ci('Animales', 'properties')
    ci_mmse = extract_ci('nps', 'mmse')
    ci_executive = extract_ci('nps','executive')
    ci_brain = extract_ci('brain', 'norm_brain_lit')
    ci_conn = extract_ci('connectivity', 'networks')
    
    # Compute lower and upper errors
    ci_lowers = means - np.array([ci_animales[0], ci_mmse[0], ci_executive[0],ci_brain[0],ci_conn[0]])
    ci_uppers = np.array([ci_animales[1], ci_mmse[1], ci_executive[1],ci_brain[1],ci_conn[1]]) - means
    
    # Combine into (2, N) array
    yerr = np.vstack([ci_lowers, ci_uppers])
    
    plt.figure()

    sns.violinplot(data=data_to_plot,x='dimension',y=metric, color="#1f77b4",order=['Fluency\n classifier', 'Screening\n classifier', 'Executive\n classifier','Structural\n classifier','Functional\n classifier'],inner=None)
    plt.errorbar(
    x=np.arange(len(data_to_plot['dimension'].unique())),
    y=means,
    yerr= yerr,
    fmt="D", color="black", capsize=5, markersize=8
    )

    plt.ylabel(metric.replace('_', ' ').upper())
    #plt.title(f"{metric.replace('_', ' ').upper()} Distribution for {model_name}")
    plt.xlabel('')
    #plt.ylim([0.6,1])
    plt.tight_layout()

    filename_to_save = f'violin_{metric}_best_models_{y_label}_{stat_folder}_{scoring}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffle'
    if not hyp_opt:
        filename_to_save = filename_to_save.replace('_hyp_opt','')
    if not feature_selection:
        filename_to_save = filename_to_save.replace('_feature_selection','')
    if not shuffle_labels:
        filename_to_save = filename_to_save.replace('_shuffle','')

    plt.savefig(Path(results_dir,'plots','bayes',f'{filename_to_save}.png'),dpi=300)
    plt.savefig(Path(results_dir,'plots','bayes',f'{filename_to_save}.svg'),dpi=300)
    plt.close()