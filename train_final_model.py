import sys, itertools, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from pingouin import partial_corr
from sklearn.model_selection import StratifiedKFold, KFold
from expected_cost.calibration import calibration_train_on_test
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal
from expected_cost.ec import CostMatrix

import pickle

import utils

correction = 'fdr_bh'

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config['project_name']
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
calibrate = config['calibrate']
avoid_stats = config['avoid_stats']
stat_folder = config['stat_folder']
hyp_opt = bool(config['n_iter'] > 0)
regress_out = bool(config['regress_out'])
feature_selection = bool(config['n_iter_features'] > 0)
filter_outliers = config['filter_outliers']
n_models = int(config['n_models'])
n_boot = int(config['n_boot'])
early_fusion = bool(config['early_fusion'])
problem_type = config['problem_type']
overwrite = bool(config['overwrite'])
parallel = bool(config['parallel'])
id_col = config['id_col']

if calibrate:
    calmethod = AffineCalLogLoss
    calparams = {'bias': True, 'priors': None}
else:
    calmethod = None
    calparams = None

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path('D:/CNC_Audio/gonza/results', project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
#tasks = main_config['tasks'][project_name]
tasks = ['craft__fugu__lamina2']
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
thresholds = main_config['thresholds'][project_name]
try:
    covars = main_config['covars'][project_name] if problem_type == 'reg' else []
except:
    covars = []
    
data_dir = str(results_dir).replace('results','data')

covariates = pd.read_csv(Path(data_dir,'all_data.csv'))[[id_col] + covars]

problem_type = config['problem_type']

scoring_metrics = ['roc_auc'] if problem_type == 'clf' else ['r2']

overwrite = bool(config['overwrite'])
metrics_names = main_config["metrics_names"][problem_type]

models_dict = {
        'clf': {
            'lr': LogisticRegression,
            'svc': SVC,
            'knnc': KNeighborsClassifier,
            'xgb': XGBClassifier,
            'lda': LDA,
            'qda': QDA,
            'nb':GaussianNB
        },
        'reg': {
            'lasso': Lasso,
            'ridge': Ridge,
            'elastic': ElasticNet,
            'svr': SVR,
            'xgb': XGBRegressor,
            'knnr': KNeighborsRegressor
        }
    }

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)
pearsons_results = pd.DataFrame(columns=['task','dimension','y_label','set','random_seed','model_type','r','p_value_corrected','p_value','method','n','95_ci','covars','correction_method'])

for scoring in scoring_metrics:    
    extremo = 1 if any(x in scoring for x in ['norm','error']) else 0
    ascending = True if extremo == 1 else False

    best_models_file = f'best_best_models_{scoring.replace("_score","")}_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled_calibrated.csv'.replace('__','_')
    if not hyp_opt:
        best_models_file = best_models_file.replace('_hyp_opt','')
    if not feature_selection:
        best_models_file = best_models_file.replace('_feature_selection','')
    if not shuffle_labels:
        best_models_file = best_models_file.replace('_shuffled','')
    if not calibrate:
        best_models_file = best_models_file.replace('_calibrated','')
        
    best_models = pd.read_csv(Path(results_dir,best_models_file))
    best_models['random_seed_test'] = best_models['random_seed_test'].map(lambda x: str(x) if str(x) != 'nan' else '') 
    #tasks = best_models['task'].unique()
    #tasks = ['craft__fugu']
    y_labels = best_models['y_label'].unique()
    #dimensions = best_models['dimension'].unique()
    dimensions = ['mlu__universal_dependencies__verbosity__word_properties__talking_intervals']
    for task,dimension,y_label in itertools.product(tasks,dimensions,y_labels):
        print(task,dimension,y_label)
        path = Path(results_dir,task,dimension,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','regress_out' if regress_out else '', 'shuffle' if shuffle_labels else '')
        
        if not Path(path).exists():
            continue
        
        random_seeds_test = [folder.name for folder in Path(path).iterdir() if folder.is_dir() and 'random_seed' in folder.name]

        random_seeds_test.append('')

        for random_seed_test in random_seeds_test:
            path_to_data = Path(path,random_seed_test)
            
            #if Path(results_dir,f'final_models',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','final_model.pkl').exists() and not overwrite:
            #    print(f'Final model already exists for {task} {dimension} {y_label} {random_seed_test}. Skipping.')
            #    continue

            try:
                model_type = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['random_seed_test'] == random_seed_test) & (best_models['y_label'] == y_label)]['model_type'].values[0] if random_seed_test != '' else best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]['model_type'].values[0]
            except:
                continue
            print(model_type)
            model_index = int(best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['random_seed_test'] == random_seed_test) & (best_models['y_label'] == y_label)]['model_index'].values[0]) if random_seed_test != '' else int(best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]['model_index'].values[0])

            if not Path(path,random_seed_test,f'all_models_{model_type}_dev_{config["bootstrap_method"]}.csv' if config['n_models'] == 0 else f'best_models_{model_type}_dev_{config["bootstrap_method"]}_{scoring}.csv' ).exists():
                continue

            all_models = pd.read_csv(Path(path,random_seed_test,f'all_models_{model_type}_dev_{config["bootstrap_method"]}.csv' if config['n_models'] == 0 else f'best_models_{model_type}_dev_{config["bootstrap_method"]}_{scoring}.csv' ))
            scoring_col = f'{scoring}_extremo'

            try:
                all_models[scoring_col] = best_models[f'{scoring}_dev'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
            except:
                all_models[scoring_col] = best_models[f'{scoring}_score_dev'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))

            best_model = all_models.sort_values(by=scoring_col,ascending=ascending).head(1)

            all_features = [col for col in best_model.columns if any(f'{x}__{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__'))) or col =='group']
            features = [col for col in all_features if best_model[col].values[0] == 1]
            params = [col for col in best_model.columns if all(x not in col for x in  all_features + metrics_names + [y_label,id_col,'Unnamed: 0','threshold','index',f'{scoring}_extremo','bootstrap_method'])]

            params_dict = {param:best_model[param].values[0] for param in params if str(best_model[param].values[0]) != 'nan'}

            if 'gamma' in params_dict.keys():
                try: 
                    params_dict['gamma'] = float(params_dict['gamma'])
                except:
                    pass

            if 'random_state' in params_dict.keys():
                params_dict['random_state'] = int(params_dict['random_state'])
            
            try:
                model = utils.Model(models_dict[problem_type][model_type](**params_dict),StandardScaler,KNNImputer,calmethod,calparams)
            except:
                params_dict = {param:best_model[param].values[0] for param in params if str(best_model[param].values[0]) != 'nan'}
                model = utils.Model(models_dict[problem_type][model_type](**params_dict),StandardScaler,KNNImputer,calmethod,calparams)
            
            X_dev = pickle.load(open(Path(path_to_data,'X_dev.pkl'),'rb'))
            y_dev = pickle.load(open(Path(path_to_data,'y_dev.pkl'),'rb'))
            if not Path(path_to_data,f'outputs_{model_type}.pkl').exists():   
                continue

            outputs_dev = pickle.load(open(Path(path_to_data,f'outputs_{model_type}.pkl'),'rb'))[:,model_index]

            if calibrate:
                scores = np.concatenate([outputs_dev[j,r] for j in range(outputs_dev.shape[0]) for r in range(outputs_dev.shape[1])])
                targets = np.concatenate([y_dev[j,r] for j in range(y_dev.shape[0]) for r in range(y_dev.shape[1])])

                _,calmodel = calibration_train_on_test(scores,targets,calmethod,calparams,return_model=True)
            if not isinstance(X_dev,pd.DataFrame):
                X_dev = pd.DataFrame(X_dev[0,0],columns=all_features)
            model.train(X_dev[features],y_dev[0,0])

            trained_model = model.model
            scaler = model.scaler
            imputer = model.imputer

            feature_importance_file = f'feature_importance_{model_type}_shuffled_calibrated.csv'.replace('__','_')

            if not shuffle_labels:
                feature_importance_file = feature_importance_file.replace('_shuffled','')
            if not calibrate:
                feature_importance_file = feature_importance_file.replace('_calibrated','')

            Path(results_dir,f'feature_importance',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,).mkdir(parents=True,exist_ok=True)
            Path(results_dir,f'final_model',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,).mkdir(parents=True,exist_ok=True)
            
            with open(Path(results_dir,f'final_model',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,'final_model.pkl'),'wb') as f:
                pickle.dump(trained_model,f)
            with open(Path(results_dir,f'final_model',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'scaler.pkl'),'wb') as f:
                pickle.dump(scaler,f)
            with open(Path(results_dir,f'final_model',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'imputer.pkl'),'wb') as f:
                pickle.dump(imputer,f)
            if calibrate:
                with open(Path(results_dir,f'final_model',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'calmodel.pkl'),'wb') as f:
                    pickle.dump(calmodel,f)
            
            if model_type == 'svc':
                model.model.kernel = 'linear'

            model.train(X_dev[features],y_dev[0,0])
            '''
            n_folds = kfold_folder.split('_')[0]

            iterator = StratifiedKFold(n_splits=int(n_folds),shuffle=True,random_state=42) if problem_type == 'clf' else KFold(n_splits=int(n_folds),shuffle=True,random_state=42)
            best_features, best_score = utils.rfe(model,X_dev[features],y_dev[0,0].values if isinstance(y_dev,pd.Series) else y_dev[0,0],None,scoring,problem_type,cmatrix=CostMatrix.zero_one_costs(K=outputs_dev.shape[-1]),priors=None,threshold=None,round_values=False)
            
            '''
            if hasattr(model.model,'feature_importance'):
                feature_importance = model.model.feature_importance
                feature_importance = pd.DataFrame({'feature':features,'importance':feature_importance}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(results_dir,f'feature_importance',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,feature_importance_file),index=False)
            elif hasattr(model.model,'coef_'):
                feature_importance = np.abs(model.model.coef_[0])
                coef = pd.DataFrame({'feature':features,'importance':feature_importance / np.sum(feature_importance)}).sort_values('importance',ascending=False)
                coef.to_csv(Path(results_dir,f'feature_importance',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,feature_importance_file),index=False)
            elif hasattr(model.model,'get_booster'):

                feature_importance = pd.DataFrame({'feature':features,'importance':model.model.feature_importances_}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(results_dir,f'feature_importance',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,feature_importance_file),index=False)
            else:
                print(task,dimension,f'No feature importance available for {model_type}')

            if problem_type == 'reg':
                sns.set_theme(style="whitegrid")  # Fondo blanco con grid sutil
                plt.rcParams.update({
                "font.family": "Arial",
                "axes.titlesize": 20,
                "axes.labelsize": 24,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
                "legend.fontsize":40
                })
                
                Path(results_dir,f'plots',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test).mkdir(parents=True,exist_ok=True)
                for set_ in ['dev','test']:
                    IDs = pickle.load(open(Path(path_to_data,f'IDs_{set_}.pkl'),'rb'))
                    
                    try:
                        outputs = pickle.load(open(Path(path_to_data,f'outputs_{model_type}.pkl'),'rb'))[:,model_index] if set_ == 'dev' else pickle.load(open(Path(path_to_data,f'outputs_test_{model_type}.pkl'),'rb'))[model_index]
                    except:
                        continue
                    y = pickle.load(open(Path(path_to_data,f'y_{set_}.pkl'),'rb'))

                    if isinstance(IDs,pd.Series):
                        IDs = IDs.values
                    if isinstance(y,pd.Series):
                        y = y.values
                    if isinstance(outputs,pd.Series):
                        outputs = outputs.values
                        
                    predictions = pd.DataFrame({'ID':IDs.flatten(),'y_pred':outputs.flatten(),'y_true':y.flatten()})
                    predictions = predictions.drop_duplicates('ID')

                    with open(Path(results_dir,f'final_model',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'predictions_dev.pkl'),'wb') as f:
                        pickle.dump(predictions,f)
                    if 'ID' in predictions.columns:
                        predictions[id_col] = predictions.pop('ID')

                    predictions.to_csv(Path(results_dir,f'final_model',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'predictions_dev.csv'),index=False)

                    if not covariates.empty:
                        predictions = pd.merge(predictions,covariates,on=id_col,how='inner')
                    
                    if 'sex' in predictions.columns:
                        #Convert sex to 0/1
                        predictions['sex'] = predictions['sex'].apply(lambda x: 1 if str(x).lower() in ['m','male','hombre','h','1'] else 0)
                    
                    method = 'pearson'

                    if len(covars) != 0: 
                        results = partial_corr(data=predictions,x='y_pred',y='y_true',covar=covars,method=method)
                        n, r, ci, p = results.loc[method,'n'], results.loc[method,'r'], results.loc[method,'CI95%'], results.loc[method,'p-val']
                    else:
                        r, p = pearsonr(predictions['y_pred'], predictions['y_true'])
                        ci = np.nan 
                        r, p = pearsonr(predictions['y_true'], predictions['y_pred'])
                        n = predictions.shape[0]
                        
                    plt.figure(figsize=(8, 6))
                    ax = sns.regplot(
                        x='y_pred', y='y_true', data=predictions,
                        scatter_kws={'alpha': 0.6, 's': 50, 'color': "#0f51df"},  # color base
                        line_kws={'color': 'black', 'linewidth': 2}
                    )

                    plt.xlabel('Predicted Value')
                    plt.ylabel('True Value')
                    plt.text(0.05, 0.95,
                            f'$r$ = {r:.2f}\n$p$ = {np.round(p,2) if p > .001 else "< .001"}',
                            fontsize=20,
                            transform=plt.gca().transAxes,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                    for spine in ax.spines.values():
                            spine.set_edgecolor('#000000')
                            spine.set_linewidth(1)

                    plt.title(f'{task} | {y_label}', pad=10)

                    plt.tight_layout()
                    plt.grid(False)

                    # Guardar resultado y cerrar

                    pearsons_results.loc[len(pearsons_results)] = [task, dimension, y_label, set_, random_seed_test,model_type, r, np.nan,p, method, n, ci, ','.join(covars) if len(covars) > 0 else 'None', np.nan]

                    save_path = Path(results_dir, f'plots', task, dimension, y_label,
                                    stat_folder, scoring,config["bootstrap_method"],
                                    'hyp_opt' if hyp_opt else '',
                                    'feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,
                                    f'{model_type}_{set_}.png')
                    
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300)
                    plt.close()

    if problem_type == 'reg':
        pearsons_results_file = f'pearons_results_{scoring}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled.csv'.replace('__','_')
        if not hyp_opt:
            pearsons_results_file = pearsons_results_file.replace('_hyp_opt','')
        if not feature_selection:
            pearsons_results_file = pearsons_results_file.replace('_feature_selection','')
        if not shuffle_labels:
            pearsons_results_file = pearsons_results_file.replace('_shuffled','')
        if len(covars) != 0:
            pearsons_results_file = pearsons_results_file.replace('.csv',f'_covars_{"_".join(covars)}.csv')
            
        p_vals = pearsons_results['p_value'].values
        reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method=correction)
        pearsons_results['p_value_corrected'] = p_vals_corrected
        pearsons_results['correction_method'] = correction

        pearsons_results.to_csv(Path(results_dir,pearsons_results_file),index=False)