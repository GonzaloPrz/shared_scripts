import pandas as pd
import numpy as np
from pathlib import Path
import itertools,sys,os,json

def new_best(current_best,value,ascending):
    if ascending:
        return value < current_best
    else:
        return value > current_best

##---------------------------------PARAMETERS---------------------------------##
bayesian = False

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']

hyp_opt = True if config["n_iter"] > 0 else False
feature_selection = True if config["n_iter_features"] > 0 else False

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]
models = main_config["models"][project_name]
metrics_names = main_config["metrics_names"][problem_type] 

pd.options.mode.copy_on_write = True 

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:/','CNC_Audio','gonza','results',project_name)

best_models = pd.DataFrame(columns=['task','dimension','y_label','model_type'] + [f'{metric}_mean_dev' for metric in metrics_names] 
                           + [f'{metric}_ic_dev' for metric in metrics_names] 
                           + [f'{metric}_mean_holdout' for metric in metrics_names]
                           + [f'{metric}_ic_holdout' for metric in metrics_names])

for scoring,task in itertools.product(scoring_metrics,tasks):
    
    extremo = 'sup' if any(x in scoring for x in ['error','norm']) else 'inf'
    ascending = True if extremo == 'sup' else False

    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        print(task,dimension)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder)

        if not path.exists():
            continue
        
        for y_label in y_labels:
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
            if not path.exists():
                continue
            random_seeds_test = [folder.name for folder in path.iterdir() if folder.is_dir() if 'random_seed' in folder.name]
            if len(random_seeds_test) == 0:
                random_seeds_test = ['']

            for random_seed_test in random_seeds_test:
                late_fusion_folders = ['']
                if Path(path,random_seed_test,'late_fusion').exists():
                    late_fusion_folders.append('late_fusion')

                for late_fusion_folder in late_fusion_folders:
                    if late_fusion_folder == 'late_fusion':
                        fusion = 'late_fusion'
                    elif '__' in dimension:
                        fusion = 'early_fusion'
                    else:
                        fusion = 'single dimension'
                    
                    files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if f'all_models_' in file.stem and 'test' in file.stem] 
                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if f'all_models_' in file.stem and 'dev' in file.stem and 'bca' in file.stem]

                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if f'best_models_' in file.stem and 'test' in file.stem and scoring in file.stem] 

                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if f'best_models_' in file.stem and 'dev' in file.stem and 'bca' in file.stem and scoring in file.stem] 
                    
                    if len(files) == 0:
                        continue
                    
                    for file in files:
                        
                        df = pd.read_csv(file)
                        
                        if f'{extremo}_{scoring}_dev' in df.columns:
                            scoring_col = f'{extremo}_{scoring}_dev'
                        else:
                            scoring_col = f'{scoring}_{extremo}'

                        df = df.sort_values(by=scoring_col,ascending=ascending)
                        
                        print(f'{file.stem.split("_")[2]}:{df.iloc[0,:][scoring_col]}')
                        best = df.iloc[0,:]
                        
                        model_type = file.stem.split('_')[2]
                        best['y_label'] = y_label
                        best['model_type'] = model_type
                        best['random_seed_test'] = random_seed_test
                        if 'idx' in df.columns:
                            best['model_index'] = df['idx'][0]
                        else:
                            best['model_index'] = df.index[0]
                                        
                        for metric in metrics_names:
                            if f'inf_{metric}_dev' in best.index:
                                best[f'{metric}_mean_dev'] = np.round(best[f"mean_{metric}_dev"],2)
                                best[f'{metric}_ic_dev'] = f'[{np.round(best[f"inf_{metric}_dev"],2)}, {np.round(best[f"sup_{metric}_dev"],2)}]'
                            elif f'inf_{metric}' in best.index:
                                best[f'{metric}_mean_dev'] = np.round(best[f"mean_{metric}"],2)
                                best[f'{metric}_ic_dev'] = f'[{np.round(best[f"inf_{metric}"],2)}, {np.round(best[f"sup_{metric}"],2)}]'
                            elif f'{metric}_inf' in best.index:
                                best[f'{metric}_mean_dev'] = np.round(best[f"{metric}_mean"],2)
                                best[f'{metric}_ic_dev'] = f'[{np.round(best[f"{metric}_inf"],2)}, {np.round(best[f"{metric}_sup"],2)}]'
                            else:
                                best[f'{metric}_mean_dev'] = np.nan
                                best[f'{metric}_ic_dev'] = np.nan

                            best[f'{metric}_test'] = np.nan
                            try:
                                mean = np.round(best[f'mean_{metric}_test'],2)
                                inf = np.round(best[f'inf_{metric}_test'],2)
                                sup = np.round(best[f'sup_{metric}_test'],2)
                                best[f'{metric}_mean_holdout'] = mean
                                best[f'{metric}_ic_holdout'] = f'[{inf}, {sup}]'
                            except:
                                pass
                        model_type = file
                        
                        dict_append = {'task':task,'dimension':dimension,'y_label':y_label,'model_type':best['model_type']}
                        dict_append.update(dict((f'{metric}_mean_dev',best[f'{metric}_mean_dev']) for metric in metrics_names))
                        dict_append.update(dict((f'{metric}_ic_dev',best[f'{metric}_ic_dev']) for metric in metrics_names))
                        dict_append.update(dict((f'{metric}_mean_hodlout',np.nan) for metric in metrics_names))
                        dict_append.update(dict((f'{metric}_ic_holdout',np.nan) for metric in metrics_names))

                        for metric in metrics_names:
                            if f'{metric}_mean_holdout' not in best.index:
                                continue
                            dict_append[f'{metric}_mean_holdout'] = best[f'{metric}_mean_holdout']
                            dict_append[f'{metric}_ic_holdout'] = best[f'{metric}_ic_holdout']
                            
                        best_models.loc[len(best_models),:] = dict_append

filename_to_save = f'all_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_shuffled.csv'.replace('__','_')

if not hyp_opt:
    filename_to_save = filename_to_save.replace('_hyp_opt','')
if not feature_selection:
    filename_to_save = filename_to_save.replace('_feature_selection','')
if not shuffle_labels:
    filename_to_save = filename_to_save.replace('_shuffled','')

best_models = best_models.dropna(axis=1,how='all')
#best_models.dropna(subset=['model_index'],inplace=True)
best_models.to_csv(Path(results_dir,filename_to_save),index=False)
