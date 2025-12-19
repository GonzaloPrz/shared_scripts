import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import json, os
import itertools

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
calibrate = config["calibrate"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
calibrate = bool(config["calibrate"])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]

metrics_names = main_config['metrics_names'][problem_type]

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for task,y_label,scoring in itertools.product(tasks,y_labels,scoring_metrics):
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',"shuffle" if shuffle_labels else "")
        
        if not path.exists():
            continue
        
        random_seeds_test = [folder.name for folder in Path(path).iterdir() if folder.is_dir() if 'random_seed' in folder.name]
        if len(random_seeds_test) == 0:
            random_seeds_test = ['']

        for random_seed_test in random_seeds_test:
            
            path_to_data = Path(path,random_seed_test)
            if calibrate:
                files = [file for file in path_to_data.iterdir() if file.is_file() and all(x in file.stem for x in ['test',f'all_models','calibrated'])]
            else:
                files = [file for file in path_to_data.iterdir() if file.is_file() and all(x in file.stem for x in ['test',f'all_models']) and 'calibrated' not in file.stem]

            for file in files:
                best_classifiers = pd.read_csv(file)
                path_to_figures = Path(results_dir,'figures_models_dev_holdout',task,dimension,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test)
                path_to_figures.mkdir(parents=True,exist_ok=True)
                for metric in metrics_names:
                    model_name = file.stem.split('_')[2]
                    if f'mean_{metric}_dev' not in best_classifiers.columns:
                        continue

                    plt.figure(figsize=(12,8))

                    sns.scatterplot(data=best_classifiers,x=f'mean_{metric}_dev',y=f'mean_{metric}_test')
                    plt.xlabel(f'mean_{metric}_dev')
                    plt.ylabel(f'mean_{metric}_holdout')
                    plt.title(f'{model_name} dev vs test')
                    
                    plt.xlim(np.min((np.min(best_classifiers[f'mean_{metric}_dev']),np.min(best_classifiers[f'mean_{metric}_test']))),np.max((np.max(best_classifiers[f'mean_{metric}_dev']),np.max(best_classifiers[f'mean_{metric}_test']))))
                    plt.ylim(np.min((np.min(best_classifiers[f'mean_{metric}_dev']),np.min(best_classifiers[f'mean_{metric}_test']))),np.max((np.max(best_classifiers[f'mean_{metric}_dev']),np.max(best_classifiers[f'mean_{metric}_test']))))

                    #Add y=x line to plot for reference
                    plt.plot([0, np.max((np.max(best_classifiers[f'mean_{metric}_dev']),np.max(best_classifiers[f'mean_{metric}_test'])))],
                             [0, np.max((np.max(best_classifiers[f'mean_{metric}_dev']),np.max(best_classifiers[f'mean_{metric}_test'])))],
                            transform=plt.gca().transAxes, ls="--", c="red")
                    
                    plt.savefig(Path(path_to_figures,f'mean_{metric}_{model_name}_calibrated.png')) if calibrate else plt.savefig(Path(path_to_figures,f'mean_{metric}_{model_name}.png'))  
                    plt.close()

                    plt.figure(figsize=(12,8))

                    sns.scatterplot(data=best_classifiers,x=f'inf_{metric}_dev',y=f'inf_{metric}_test')
                    plt.xlabel(f'inf {metric} dev')
                    plt.ylabel(f'inf {metric} holdout')
                    plt.title(f'{model_name} dev vs test')
                    
                    #Add diagonal line to plot for reference
                    plt.plot([0, np.max((np.max(best_classifiers[f'inf_{metric}_dev']),np.max(best_classifiers[f'inf_{metric}_test'])))],
                            [0, np.max((np.max(best_classifiers[f'inf_{metric}_dev']),np.max(best_classifiers[f'inf_{metric}_test'])))],
                            transform=plt.gca().transAxes, ls="--", c="red")
                    
                    plt.xlim(np.min((np.min(best_classifiers[f'inf_{metric}_dev']),np.min(best_classifiers[f'inf_{metric}_test']))),np.max((np.max(best_classifiers[f'inf_{metric}_dev']),np.max(best_classifiers[f'inf_{metric}_test']))))
                    plt.ylim(np.min((np.min(best_classifiers[f'inf_{metric}_dev']),np.min(best_classifiers[f'inf_{metric}_test']))),np.max((np.max(best_classifiers[f'inf_{metric}_dev']),np.max(best_classifiers[f'inf_{metric}_test']))))

                    plt.savefig(Path(path_to_figures,f'inf_{metric}_{model_name}_calibrated.png')) if calibrate else plt.savefig(Path(path_to_figures,f'inf_{metric}_{model_name}.png'))  
                    plt.close()
                    plt.figure(figsize=(12,8))

                    sns.scatterplot(data=best_classifiers,x=f'sup_{metric}_dev',y=f'sup_{metric}_test')
                    plt.xlabel(f'sup {metric} dev')
                    plt.ylabel(f'sup {metric} holdout')
                    plt.title(f'{model_name} dev vs test')
                    #Add y=x line to plot for reference
                    plt.plot([0, np.max((np.max(best_classifiers[f'sup_{metric}_dev']),np.max(best_classifiers[f'sup_{metric}_test'])))], 
                             [0, np.max((np.max(best_classifiers[f'sup_{metric}_dev']),np.max(best_classifiers[f'sup_{metric}_test'])))],
                            transform=plt.gca().transAxes, ls="--", c="red")
                    
                    plt.xlim(np.min((np.min(best_classifiers[f'sup_{metric}_dev']),np.min(best_classifiers[f'sup_{metric}_test']))),np.max((np.max(best_classifiers[f'sup_{metric}_dev']),np.max(best_classifiers[f'sup_{metric}_test']))))
                    plt.ylim(np.min((np.min(best_classifiers[f'sup_{metric}_dev']),np.min(best_classifiers[f'sup_{metric}_test']))),np.max((np.max(best_classifiers[f'sup_{metric}_dev']),np.max(best_classifiers[f'sup_{metric}_test']))))

                    plt.savefig(Path(path_to_figures,f'sup_{metric}_{model_name}_calibrated.png')) if calibrate else plt.savefig(Path(path_to_figures,f'sup_{metric}_{model_name}.png'))
                    plt.close()  