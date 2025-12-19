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

project_name = config['project_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
calibrate = config['calibrate']
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
parallel = config['parallel']
regress_out = config['regress_out']
hyp_opt = config['n_iter'] > 0 
feature_selection = config['n_iter_features'] > 0

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]

problem_type = config['problem_type']
scoring_metrics = ['roc_auc'] if problem_type == 'clf' else ['r2']

metrics_names = main_config['metrics_names'][problem_type] 

pd.options.mode.copy_on_write = True 

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:/','CNC_Audio','gonza','results',project_name)
for scoring in scoring_metrics:
    best_models = pd.DataFrame(columns=['task','dimension','y_label','model_type','model_index','random_seed_test','fusion'] + [f'{metric}_dev' for metric in metrics_names] + [f'{metric}_holdout' for metric in metrics_names])
    best_best_models = pd.DataFrame(columns=['task','dimension','y_label','model_type','model_index','random_seed_test','fusion'] + [f'{metric}_dev' for metric in metrics_names] + [f'{metric}_holdout' for metric in metrics_names])
    
    for task in tasks:

        extremo = 1 if any(x in scoring for x in ['error','norm']) else 0
        ascending = any(x in scoring for x in ['error','norm'])

        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,kfold_folder)

            if not path.exists():
                continue
            
            for y_label in y_labels:
                path = Path(results_dir,task,dimension,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','regress_out' if regress_out else '','shuffle' if shuffle_labels else '')
                if not path.exists():
                    continue
                random_seeds_test = [folder.name for folder in path.iterdir() if folder.is_dir() if 'random_seed' in folder.name]
                
                random_seeds_test.append('')

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
                        
                        if calibrate:
                            files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['all_models_',f'test_{config["bootstrap_method"]}','calibrated'])] 
                            if len(files) == 0:
                                files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['all_models_',f'dev_{config["bootstrap_method"]}','calibrated'])]
                            if len(files) == 0:
                                files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['best_models_',f'test_{config["bootstrap_method"]}','calibrated'])] 
                            if len(files) == 0:
                                files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['best_models_',f'dev_{config["bootstrap_method"]}','calibrated'])] 
                        else:
                            files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['all_models_',f'test_{config["bootstrap_method"]}']) and 'calibrated' not in file.stem] 
                            if len(files) == 0:
                                files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['best_models_',f'test_{config["bootstrap_method"]}']) and 'calibrated' not in file.stem] 
                            if len(files) == 0:
                                files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['all_models_',f'dev_{config["bootstrap_method"]}']) and 'calibrated' not in file.stem]
                            if len(files) == 0:
                                files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['best_models_',f'dev_{config["bootstrap_method"]}']) and 'calibrated' not in file.stem] 
                            if len(files) == 0:
                                files = [file for file in Path(path,random_seed_test,late_fusion_folder).iterdir() if all(x in file.stem for x in ['all_models','dev']) and 'calibrated' not in file.stem] 
                        
                        if len(files) == 0:
                            continue
                        
                        best = None
                        for file in files:
                            if file.suffix != '.csv':
                                continue

                            df = pd.read_csv(file)
                            
                            scoring_col = f'{scoring}_extremo'
                            try:                            
                                df[scoring_col] = df[scoring].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
                            except:
                                df[scoring_col] = df[f'{scoring}_score'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))

                            df = df.sort_values(by=scoring_col,ascending=ascending)
                            
                            print(f"{file.stem.split('_')[2]}:{df.iloc[0,:][scoring_col]}")
                            
                            dict_append = {'task':task,'dimension':dimension,'y_label':y_label,'model_type':file.stem.split('_')[2],'model_index':df.index[0],'random_seed_test':random_seed_test,'fusion':fusion}

                            dict_append.update(dict((f'{metric}_dev',df.iloc[0][metric]) for metric in metrics_names))

                            dict_append.update(dict((f'{metric}_holdout',df.iloc[0][f'{metric}_holdout']) for metric in metrics_names) if all([f'{x}_holdout' in df.columns for x in metrics_names]) else dict((f'{metric}_holdout',np.nan) for metric in metrics_names))
                            best_models.loc[len(best_models),:] = dict_append

                            if best is None:
                                best = df.iloc[0,:]
                                
                                model_type = file.stem.split('_')[2]
                                best['y_label'] = y_label
                                best['model_type'] = model_type
                                best['random_seed_test'] = random_seed_test
                                if 'idx' in df.columns:
                                    best['model_index'] = df['idx'][0]
                                else:
                                    best['model_index'] = df.index[0]
                            
                                best_file = file
                            else:
                                if new_best(best[scoring_col],df.iloc[0,:][scoring_col],ascending):
                                    best = df.iloc[0,:]

                                    model_type = file.stem.split('_')[2]
                                    best['y_label'] = y_label
                                    best['model_type'] = model_type
                                    best['random_seed_test'] = random_seed_test
                                    if 'idx' in df.columns:
                                        best['model_index'] = df['idx'][0]
                                    else:
                                        best['model_index'] = df.index[0]

                                    best_file = file
                        
                        model_type = file
                        
                        dict_append = {'task':task,'dimension':dimension,'y_label':y_label,'model_type':best['model_type'],'model_index':best['model_index'],'random_seed_test':random_seed_test,'fusion':fusion}
                        dict_append.update(dict((f'{metric}_dev',best[metric]) for metric in metrics_names))

                        dict_append.update(dict((f'{metric}_holdout',best[f'{metric}_holdout']) for metric in metrics_names) if all([f'{x}_holdout' in best.keys() for x in metrics_names]) else dict((f'{metric}_holdout',np.nan) for metric in metrics_names))

                        if best_best_models.empty:
                            best_best_models = pd.DataFrame(columns=dict_append.keys())
                        
                        best_best_models.loc[len(best_best_models),:] = dict_append

    filename_to_save = f'best_models_{scoring}_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled_calibrated.csv'.replace('__','_')

    if not hyp_opt:
        filename_to_save = filename_to_save.replace('_hyp_opt','')
    if not feature_selection:
        filename_to_save = filename_to_save.replace('_feature_selection','')
    if not shuffle_labels:
        filename_to_save = filename_to_save.replace('_shuffled','')
    if not calibrate:
        filename_to_save = filename_to_save.replace('_calibrated','')

    best_models = best_models.dropna(axis=1,how='all')
    #best_models.dropna(subset=['model_index'],inplace=True)
    best_models.to_csv(Path(results_dir,filename_to_save),index=False)
    best_best_models.to_csv(Path(results_dir,f'best_{filename_to_save}'),index=False)