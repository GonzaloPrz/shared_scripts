import pickle, sys, json, os, itertools
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from expected_cost.ec import CostMatrix

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

parallel = True 
cmatrix = None

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = bool(config['n_iter'])
scoring_metrics = config['scoring_metrics']

if 'feature_selection' in config.keys():
    feature_selection = bool(config['feature_selection'])
else:
    feature_selection = bool(config['n_iter_features'])
    
filter_outliers = bool(config['filter_outliers'])
calibrate = bool(config["calibrate"])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:","CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
thresholds = main_config['thresholds'][project_name]
data_file = main_config['data_file'][project_name]

if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]
    
problem_type = main_config['problem_type'][project_name]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

##---------------------------------PARAMETERS---------------------------------##
results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for scoring in scoring_metrics:

    extremo = 1 if any(x in scoring for x in ['error','norm']) else 0
    ascending = True if any(x in scoring for x in ['error','norm']) else False

    best_models_file = f'best_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffle_calibrated.csv'.replace('__','_')

    if not feature_selection:
        best_models_file = best_models_file.replace('_feature_selection','')
    if not shuffle_labels:
        best_models_file = best_models_file.replace('_shuffle','')
    if not hyp_opt:
        best_models_file = best_models_file.replace('_hyp_opt','')
    if not calibrate:
        best_models_file = best_models_file.replace('_calibrated','')
    try:
        best_models = pd.read_csv(Path(results_dir,best_models_file))
    except:
        best_models = pd.read_csv(Path(results_dir,best_models_file.replace('.csv','_bayes.csv')))
    
    for r,row in best_models.iterrows():
        task = row.task
        dimension = row.dimension
        y_label = row.y_label
        random_seed = row.random_seed_test

        if str(random_seed) == 'nan':
            random_seed = ''

        print(task,dimension)
        best_model = best_models[(best_models['task'] == task) & (best_models['y_label'] == y_label) & (best_models['dimension'] == dimension)]
        
        scoring_col = f'{scoring}_extremo'
        if f'{scoring}_score' in best_models.columns:
            best_model[scoring_col] = best_model[f'{scoring}_score'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
        elif f'{scoring}_dev' in best_models.columns:
            best_model[scoring_col] = best_model[f'{scoring}_dev'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
        else:
            best_model[scoring_col] = best_model[scoring].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
        
        best_model = best_model.sort_values(by=scoring_col,ascending=ascending)

        model_name = best_model['model_type'].values[0]

        bayes = 'model_index' not in best_models.columns

        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'bayes' if bayes else '',scoring if bayes else '', 'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')

        with open(Path(path,random_seed,'y_dev.pkl'),'rb') as f:
            y_dev = np.array(pickle.load(f),dtype=int)

        with open(Path(path,random_seed,f'outputs_{model_name}.pkl'),'rb') as f:
            outputs_dev = pickle.load(f)
        
        if not bayes:
            model_index = best_model['model_index'].values[0]
            outputs = outputs_dev.squeeze()[model_index]
        else:
            outputs = outputs_dev.squeeze()
            
        _, y_pred_dev = utils.get_metrics_clf(outputs, y_dev, [], cmatrix) if not bayes else utils.get_metrics_clf(outputs, y_dev, [], cmatrix)
            
        try:
            cmatrix_dev = confusion_matrix(y_dev.flatten(), y_pred_dev.flatten(),normalize='true')
        except:
            continue

        if Path(path,random_seed,'y_test.pkl').exists():
            with open(Path(path,random_seed,'y_test.pkl'),'rb') as f:
                y_test = np.array(pickle.load(f),dtype=int)
            with open(Path(path,random_seed,f'outputs_test_{model_name}.pkl'),'rb') as f:
                outputs_test_ = pickle.load(f)
                if not bayes:
                    outputs_test = outputs_test_.squeeze()[model_index]
                else:
                    outputs_test = outputs_test_.squeeze()
            _, y_pred_test = utils.get_metrics_clf(outputs_test, y_test, [], cmatrix)
            cmatrix_test = confusion_matrix(y_test.flatten(), y_pred_test.flatten(),normalize='true')
            fig, ax = plt.subplots(1,2,figsize=(10,5))
            ConfusionMatrixDisplay(cmatrix_dev).plot(ax=ax[0])
            ax[0].set_title('X-val')
            ConfusionMatrixDisplay(cmatrix_test).plot(ax=ax[1])
            ax[1].set_title('Test')
        else:
            fig, ax = plt.subplots(1,1,figsize=(10,5))
            ConfusionMatrixDisplay(cmatrix_dev).plot(ax=ax)
            ax.set_title('X-val')
        #Plot confusion matrix and save as fig
        
        fig.suptitle(f'{task} - {dimension} - {y_label} - {model_name} - {random_seed}')
        plt.savefig(Path(path,random_seed,f'confusion_matrix_{model_name}_{config["bootstrap_method"]}.png'))


                

                


