import seaborn as sns
import pandas as pd
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings, json, os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import bootstrap

from expected_cost.ec import CostMatrix

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

warnings.filterwarnings('ignore')

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config['project_name']
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
test_size = float(config['test_size'])
n_boot = int(config['n_boot'])

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path('D:/CNC_Audio/gonza/results', project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

scoring_metrics = main_config['scoring_metrics'][project_name]
metrics_names = main_config['metrics_names'][main_config['problem_type'][project_name]]
tasks = main_config['tasks'][project_name]
dimensions = main_config['single_dimensions'][project_name]
y_labels = main_config['y_labels'][project_name]

cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]

# Set the style for the plots
sns.set(style='whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

for scoring in scoring_metrics:
    extremo = 1 if 'norm' in scoring else 0
    ascending = True if extremo == 1 else False
    data_to_plot = pd.DataFrame()
    best_models_filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection.csv'.replace('__','_')
    best_models = pd.read_csv(Path(results_dir,best_models_filename))

    for r, row in best_models.iterrows():
        task = row.task
        dimension = row.dimension
        y_label = row.y_label
        model_name = row.model_type

        data_append = pd.DataFrame()
        Path(results_dir,'plots').mkdir(parents=True, exist_ok=True)

        print(task, dimension)
        path_to_results = Path(results_dir, task, dimension, scaler_name, kfold_folder, y_label, stat_folder,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')

        if not hyp_opt:
            best_models_filename = best_models_filename.replace('_hyp_opt','')
        if not feature_selection:
            best_models_filename = best_models_filename.replace('_feature_selection','')
                            
        file = f'all_models_{model_name}_dev_{config["bootstrap_method"]}.csv'
        
        if config['n_models'] != 0:
            file = file.replace('all_models', 'best_models').replace('.csv', f'_{scoring}.csv')
        
        if str(row['random_seed_test']) == 'nan':
            random_seed = ''
        else:
            random_seed = row.random_seed_test

        scoring_col = f'{scoring}_extremo'

        df_filename = pd.read_csv(Path(path_to_results, random_seed, file))
        df_filename[scoring_col] = df_filename[scoring].apply(lambda x: x.split('(')[1].replace(')','').split(', ')[extremo])

        df_filename = df_filename.sort_values(by=scoring_col,ascending=ascending)

        model_index = df_filename.index[0]

        if 'threshold' in df_filename.columns:
            threshold = df_filename['threshold'][0]
        else:
            threshold = None
                            
        outputs_filename = f'outputs_{model_name}.pkl'
        outputs_ = pickle.load(open(Path(path_to_results, random_seed, outputs_filename), 'rb'))[:,model_index]
        y_true_ = pickle.load(open(Path(path_to_results,random_seed, f'y_dev.pkl'), 'rb'))
        IDs_ = pickle.load(open(Path(path_to_results,random_seed, f'IDs_dev.pkl'), 'rb'))
        
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
    data_to_plot = data_to_plot[((data_to_plot["task"] == "Animales__P") & (data_to_plot["dimension"] == "properties")) |
                                ((data_to_plot["task"] == "cog") & (data_to_plot["dimension"] == "neuropsico_digits__neuropsico_tmt")) |
                                ((data_to_plot["task"] == "brain") & (data_to_plot["dimension"] == "norm_brain_lit")) 
                                ]
    
    data_to_plot["dimension"] = data_to_plot["dimension"].map({"properties":"Speech","neuropsico_digits__neuropsico_tmt": "Cognitive","norm_brain_lit":"Brain"})
    for metric in metrics_names:
        plt.figure()
        sns.violinplot(data=data_to_plot,x='dimension',y=metric, color="#1f77b4",)
        plt.ylabel(metric.replace('_', ' ').upper())
        #plt.title(f"{metric.replace('_', ' ').upper()} Distribution for {model_name}")
        plt.tight_layout()
        filename_to_save = f'violin_{metric}_best_models_{y_label}_{stat_folder}_{scoring}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffle'
        if not hyp_opt:
            filename_to_save = filename_to_save.replace('_hyp_opt','')
        if not feature_selection:
            filename_to_save = filename_to_save.replace('_feature_selection','')
        if not shuffle_labels:
            filename_to_save = filename_to_save.replace('_shuffle','')

        plt.savefig(Path(results_dir,'plots',f'{filename_to_save}.png'))
        plt.savefig(Path(results_dir,'plots',f'{filename_to_save}.svg'))
        plt.close()