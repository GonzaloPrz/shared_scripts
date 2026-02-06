import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings, json, os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from expected_cost.utils import plot_hists
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

warnings.filterwarnings('ignore')

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config['project_name']
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = bool(config['n_iter'] > 0)
if 'feature_selection' in config.keys():
    feature_selection = bool(config['feature_selection'])
else:
    feature_selection = bool(config['n_iter_features'] > 0)

filter_outliers = bool(config['filter_outliers'])
test_size = float(config['test_size'])
n_boot = int(config['n_boot'])
calibrate = bool(config["calibrate"])
bayes = bool(config["bayes"])

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path('D:/CNC_Audio/gonza/results', project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

scoring_metrics = main_config['scoring_metrics'][project_name]
metrics_names = main_config['metrics_names'][main_config['problem_type'][project_name]]
tasks = main_config['tasks'][project_name]
y_labels = main_config['y_labels'][project_name]
data_file = main_config['data_file'][project_name]

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

if problem_type == 'clf':
    for scoring in scoring_metrics:
        extremo = 1 if any(x in scoring for x in ['norm','error']) else 0
        ascending = True if extremo == 1 else False

        best_models_filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{config["bootstrap_method"]}_{stat_folder}_hyp_opt_feature_selection_calibrated_bayes.csv'.replace('__','_')
        if not hyp_opt:
            best_models_filename = best_models_filename.replace('_hyp_opt','')
        if not feature_selection:
            best_models_filename = best_models_filename.replace('_feature_selection','')
        if not calibrate:
            best_models_filename = best_models_filename.replace('_calibrated','')
        if not bayes:
            best_models_filename = best_models_filename.replace('_bayes','')

        if not Path(results_dir,best_models_filename).exists():
            continue

        best_models = pd.read_csv(Path(results_dir,best_models_filename))
        
        for r,row in best_models.iterrows():
            task = row.task
            y_label = row.y_label
            dimension = row.dimension
            model_name = row.model_type
            random_seed = row.random_seed_test  
            
            if str(random_seed) == 'nan':
                random_seed = ''

            path_to_results = Path(results_dir,task, dimension, scaler_name, kfold_folder, y_label, stat_folder,'bayes' if bayes else '',scoring if bayes else '','hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')

            Path(results_dir,'plots',task,dimension,y_label,stat_folder,'bayes' if bayes else '',scoring if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed).mkdir(parents=True, exist_ok=True)
            
            if not bayes:
                file = f'all_models_{model_name}_dev_bca.csv'
                
                if not Path(path_to_results,file).exists():
                    continue

                if config['n_models'] != 0:
                    file = file.replace('all_models', 'best_models').replace('.csv', f'_{scoring}.csv')
                
                if not Path(path_to_results,random_seed,file).exists():
                    continue
                
                scoring_col = f'{scoring}_extremo'

                df_filename = pd.read_csv(Path(path_to_results, random_seed, file))
                df_filename[scoring_col] = df_filename[scoring].apply(lambda x: x.split('(')[1].replace(')','').split(', ')[extremo])

                df_filename = df_filename.sort_values(by=scoring_col,ascending=ascending)

                model_index = df_filename.index[0]

                if 'threshold' in df_filename.columns:
                    threshold = df_filename['threshold'][0]
                else:
                    threshold = None
                
                if Path(path_to_results, 'shuffle', random_seed,file).exists() and shuffle_labels:
                    df_filename_shuffle = pd.read_csv(Path(path_to_results, 'shuffle', random_seed, f'all_models_{model_name}_dev_bca.csv')).sort_values(f'{scoring}_{extremo}'.replace('_score',''), ascending=ascending)
                    model_index_shuffle = df_filename_shuffle.index[0]
                    if 'threshold' in df_filename_shuffle.columns:
                        threshold_shuffle = df_filename_shuffle['threshold'][0]
                    else:
                        threshold_shuffle = None
                    
            outputs_filename = f'cal_outputs_{model_name}.pkl' if calibrate else f'outputs_{model_name}.pkl'
            
            if not bayes:
                outputs_ = pickle.load(open(Path(path_to_results, random_seed, outputs_filename), 'rb'))[:,model_index]
            else:
                outputs_ = pickle.load(open(Path(path_to_results, random_seed, outputs_filename), 'rb'))

            ax = None

            if Path(path_to_results,random_seed,outputs_filename.replace('outputs','outputs_test')).exists():
                if not bayes:
                    outputs_test = pickle.load(open(Path(path_to_results,random_seed,outputs_filename.replace('outputs','outputs_test')), 'rb'))[model_index]
                else:
                    outputs_test = pickle.load(open(Path(path_to_results,random_seed,outputs_filename.replace('outputs','outputs_test')), 'rb'))
                #Add missing dimensions: model_index, j
                y_test = pickle.load(open(Path(path_to_results,random_seed,f'y_test.pkl'), 'rb'))
                IDs_test = pickle.load(open(Path(path_to_results,random_seed,f'IDs_test.pkl'), 'rb'))
                scores = outputs_test
                plot_hists(y_test, scores, outfile=Path(results_dir,'plots',task,dimension,y_label,stat_folder,'bayes' if bayes else '',scoring if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed,f'best_{model_name}_cal_logpost.png' if calibrate else '' + f'best_{model_name}_logpost_test.png'), nbins=50, group_by='score', style='--', label_prefix='test ', axs=None)
            
            #Add missing dimensions: model_index, j            
            y_true_ = pickle.load(open(Path(path_to_results,random_seed, f'y_dev.pkl'), 'rb'))
            while y_true_.ndim < 3:
                y_true_ = np.expand_dims(y_true_, axis=0)
            
            while outputs_.ndim < 4:
                outputs_ = np.expand_dims(outputs_, axis=0)

            IDs = pickle.load(open(Path(path_to_results,random_seed, f'IDs_dev.pkl'), 'rb'))
            
            scores = np.concatenate([outputs_[0,r] for r in range(outputs_.shape[1])])
            y_true = np.concatenate([y_true_[0,r] for r in range(y_true_.shape[1])])
            
            plot_hists(y_true, scores, outfile=Path(results_dir,'plots',task,dimension,y_label,stat_folder,'bayes' if bayes else '',scoring if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed,f'best_{model_name}_cal_logpost.png' if calibrate else '' + f'best_{model_name}_logpost.png'), nbins=50, group_by='score', style='-', label_prefix='dev ', axs=ax)
            