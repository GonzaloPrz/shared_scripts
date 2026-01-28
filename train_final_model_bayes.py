import sys, itertools, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier as xgboost
from xgboost import XGBRegressor as xgboostr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, LeaveOneGroupOut, StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.multitest import multipletests
from pingouin import partial_corr
from scipy.stats import shapiro
import shap

import pickle

import utils

from expected_cost.ec import CostMatrix

config = json.load(Path(Path(__file__).parent,'config.json').open())
project_name = config["project_name"]
scaler_name = config['scaler_name']
n_folds = config['n_folds_inner']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = config['n_iter'] > 0
feature_selection = bool(config['feature_selection'])
calibrate = bool(config["calibrate"])
n_iter = int(config["n_iter"])
init_points = int(config["init_points"])
id_col = config['id_col']
scoring = config['scoring_metric']
problem_type = config['problem_type']
overwrite = bool(config["overwrite"])
filter_outliers = bool(config['filter_outliers']) if problem_type == 'reg' else False
round_values = bool(config['round_values'])
cut_values = bool(config['cut_values'] > 0)
regress_out = len(config['covariates']) > 0 if problem_type == 'reg' else False

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

data_file = main_config['data_file'][project_name]

try:
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
except KeyError:
    cmatrix = None

try:
    thresholds = main_config['thresholds'][project_name]
except KeyError:
    thresholds =[None]

try:
    covars = main_config['covars'][project_name] if problem_type == 'reg' else []
except:
    covars = []

models_dict = {'clf':{
                    'lr':LR,
                    'knnc':KNNC,
                    'xgb':xgboost,
                    'rf':RandomForestClassifier,
                    'svc':SVC,
                    'qda':QDA,
                    'lda': LDA
                    },
                
                'reg':{'lasso':Lasso,
                    'ridge':Ridge,
                    'elastic':ElasticNet,
                    'rf':RandomForestRegressor,
                    'knnr':KNNR,
                    'svr':SVR,
                    'xgb':xgboostr
                    }
}

hyperp = json.load(Path(Path(__file__).parent,'hyperparameters.json').open())

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)
data_dir = str(results_dir).replace('results','data')
corr_results = pd.DataFrame(columns=['r','p_value_corrected','p_value','method','n','95_ci','covars','correction_method'])

correction = 'fdr_bh'

covariates = pd.read_csv(Path(data_dir,data_file))[[id_col]+covars]

method = 'pearson'

for threshold in thresholds:
    if str(threshold) == 'None':
        threshold = None

    ending = f'_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_filter_outliers_round_cut_shuffled_calibrated_bayes_{config["version"]}.csv'.replace('__','_')
    if not hyp_opt:
        ending = ending.replace('_hyp_opt','')
    if not feature_selection:
        ending = ending.replace('_feature_selection','')
    if not filter_outliers:
        ending = ending.replace('_filter_outliers','')
    if not round_values:
        ending = ending.replace('_round','')
    if not cut_values:
        ending = ending.replace('_cut','')
    if not shuffle_labels:
        ending = ending.replace('_shuffled','')
    if not calibrate:
        ending = ending.replace('_calibrated','')

    filenames = [file.name for file in Path(results_dir).iterdir() if file.name.startswith('best_best_models') and file.name.endswith(ending)]

    for filename in filenames:
        scoring = filename.replace('best_best_models_','').replace(ending,'')

        best_models = pd.read_csv(Path(results_dir,filename))

        for r, row in best_models.iterrows():
            task = row.task
            dimension = row.dimension
            y_label = row.y_label
            model_type = row.model_type

            print(task,dimension,y_label,model_type)
            path_to_results_ = Path(results_dir,task,dimension,kfold_folder,y_label,stat_folder)
            
            path_to_results = Path(path_to_results_,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'])
            random_seeds = [folder.name for folder in path_to_results.iterdir() if 'random_seed' in folder.name]

            if len(random_seeds) == 0:
                random_seeds = [''] 

            for random_seed in random_seeds:
                try:
                    y_dev = np.load(open(Path(path_to_results,random_seed,'y_dev.npy'),'rb'),allow_pickle=True)
                    X_train = np.load(open(Path(path_to_results,random_seed,'X_train.npy'),'rb'),allow_pickle=True)
                    y_train = np.load(open(Path(path_to_results,random_seed,'y_train.npy'),'rb'),allow_pickle=True)
                    outputs_dev = np.load(open(Path(path_to_results,random_seed,f'outputs_{model_type}.npy'),'rb'),allow_pickle=True)
                    
                    IDs = np.load(open(Path(path_to_results,random_seed,'IDs_dev.npy'),'rb'),allow_pickle=True)
                    IDs_train = np.load(open(Path(path_to_results,random_seed,'IDs_train.npy'),'rb'),allow_pickle=True)
                except:
                    y_dev = pickle.load(open(Path(path_to_results,random_seed,'y_dev.pkl'),'rb'))
                    X_train = pickle.load(open(Path(path_to_results,random_seed,'X_train.pkl'),'rb'))
                    y_train = pickle.load(open(Path(path_to_results,random_seed,'y_train.pkl'),'rb'))
                    outputs_dev = pickle.load(open(Path(path_to_results,random_seed,f'outputs_{model_type}.pkl'),'rb'))
                    
                    IDs = pickle.load(open(Path(path_to_results,random_seed,'IDs_dev.pkl'),'rb'))
                    IDs_train = pickle.load(open(Path(path_to_results,random_seed,'IDs_train.pkl'),'rb'))
                
                if problem_type == 'reg':
                    y_pred = np.round(outputs_dev,decimals=0) if config['round_values'] else outputs_dev
                    predictions = pd.DataFrame({'id':IDs.flatten(),'y_pred':y_pred.flatten(),'y_true':y_dev.flatten()})
                else:
                    _, y_pred = utils.get_metrics_clf(outputs_dev, y_dev, [], cmatrix=cmatrix, priors=None, threshold=threshold)
                    predictions = {'id':IDs.flatten(),'y_pred':y_pred.flatten(),'y_true':y_dev.flatten()}
                    for c in range(outputs_dev.shape[-1]):
                        predictions[f'outputs_class_{c}'] = outputs_dev[:,:,c].flatten()
                    predictions = pd.DataFrame(predictions)
                    
                predictions = predictions.drop_duplicates('id')

                Path(results_dir,f'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '', config['version'],random_seed).mkdir(exist_ok=True,parents=True)

                with open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,f'predictions_dev.npy'),'wb') as f:
                    pickle.dump(predictions,f)
                predictions.to_csv(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,f'predictions_dev.csv'),index=False)

                if problem_type == 'reg':
                    sns.set_theme(style="whitegrid")  # Fondo blanco con grid sutil
                    plt.rcParams.update({
                        "font.family": "Arial",
                        "axes.titlesize": 26,
                        "axes.labelsize": 20,
                        "xtick.labelsize": 20,
                        "ytick.labelsize": 20,
                        })

                    if not covariates.empty:
                        predictions = pd.merge(predictions,covariates,on=id_col,how='inner')
                    
                    
                    #if all([shapiro(predictions['y_pred'])[1] > 0.1,shapiro(predictions['y_true'])[1] > 0.1]):
                    #    method = 'pearson'
                    #else:
                    #    method = 'spearman'
                    
                    if len(covars) != 0: 
                        results = partial_corr(data=predictions,x='y_pred',y='y_true',covar=covars,method=method)
                        n, r, ci, p = results.loc[method,'n'], results.loc[method,'r'], results.loc[method,'CI95%'], results.loc[method,'p-val']
                    else:
                        r, p = pearsonr(predictions['y_pred'], predictions['y_true']) if method == 'pearson' else spearmanr(predictions['y_pred'], predictions['y_true'])
                        n = predictions.shape[0]
                        ci = np.nan

                    # Guardar resultado y cerrar
                    corr_results.loc[len(corr_results)] = [r, '',p, method,n, str(ci),str(covars),np.nan]

                    save_path = Path(results_dir, f'plots', task, dimension, y_label,
                                    stat_folder,config["bootstrap_method"],'bayes',scoring,
                                    'hyp_opt' if hyp_opt else '',
                                    'feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,
                                    f'{model_type}_{method}.png')
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    plt.figure(figsize=(8, 6))
                    sns.regplot(
                        x='y_pred', y='y_true', data=predictions,
                        scatter_kws={'alpha': 0.6, 's': 50, 'color': '#c9a400'},  # color base
                        line_kws={'color': 'black', 'linewidth': 2}
                    )

                    plt.xlabel('Predicted Value')
                    plt.ylabel('True Value')

                    plt.text(0.05, 0.95,
                            f'$r$ = {r:.2f}\n$p$ = {np.round(p,3) if p > .001 else "< .001"}',
                            fontsize=20,
                            transform=plt.gca().transAxes,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                    plt.title(f'{task} | {y_label}', fontsize=25, pad=15)

                    plt.tight_layout()
                    plt.grid(False)
                    try:
                        plt.savefig(save_path, dpi=300)
                    except:
                        continue
                    plt.close()

                if Path(results_dir,f'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','rounded' if round_values else '','cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,f'model_{model_type}.npy').exists() and not overwrite:
                    print('Model already exists')
                    continue
                
                if not Path(path_to_results,random_seed,f'all_models_{model_type}.csv').exists():
                    continue

                all_models = pd.read_csv(Path(path_to_results,random_seed,f'all_models_{model_type}.csv'))
                
                features = [col for col in all_models.columns if any(f'{single_task}__' in col for single_task in task.split('__'))]
                
                if not isinstance(X_train, pd.DataFrame):
                    X_train = pd.DataFrame(X_train, columns=features)
                
                if n_folds == 0:
                    n_folds = int(np.floor(X_train.shape[0]/np.unique(y_train).shape[0]))
                    n_max = X_train.shape[0] - np.unique(y_train).shape[0]
                    CV = (StratifiedGroupKFold(n_splits=n_folds, shuffle=True)
                                if config['stratify'] and problem_type == 'clf'
                                else GroupKFold(n_splits=n_folds, shuffle=True))   
                elif n_folds == -1:
                    CV = LeaveOneGroupOut()
                    n_max = X_train.shape[0] - 1
                elif n_folds < 1:
                    CV = (StratifiedShuffleSplit(n_splits=1,test_size=n_folds)
                                if config['stratify'] and problem_type == 'clf'
                                else GroupShuffleSplit(n_splits=1,test_size=n_folds))
                    n_max = int(X_train.shape[0]*(1-n_folds))
                else:
                    CV = (StratifiedGroupKFold(n_splits=int(n_folds), shuffle=True)
                                if config['stratify'] and problem_type == 'clf'
                                else GroupKFold(n_splits=int(n_folds), shuffle=True))  
                    n_max = int(X_train.shape[0]*(1-1/n_folds))
                        
                hyperp['knnc']['n_neighbors'] = (1,n_max)
                hyperp['knnr']['n_neighbors'] = (1,n_max)

                model_class = models_dict[problem_type][model_type]
                scaler = StandardScaler if scaler_name == 'StandardScaler' else MinMaxScaler
                imputer = KNNImputer
                if cmatrix is None and problem_type == 'clf':
                    cmatrix = CostMatrix.zero_one_costs(K=len(np.unique(y_train)))
                
                if int(config["n_iter"]):
                    try:
                        best_params, best_score = utils.tuning(model_class,scaler,imputer,X_train,y_train.values if isinstance(y_train,pd.Series) else y_train,IDs_train,hyperp[model_type],CV,init_points=int(config['init_points']),n_iter=n_iter,scoring=scoring,problem_type=problem_type,cmatrix=cmatrix,priors=None,threshold=threshold,calmethod=None,calparams=None,round_values=config['round_values'],covariates=covariates if regress_out else None,fill_na=config['fill_na'],regress_out_method=config['regress_out_method'])
                    except Exception as e:
                        print(e)
                        continue
                else: 
                    best_params = model_class().get_params()

                if problem_type == 'clf' and model_class == SVC:
                    best_params['probability'] = True
                
                if 'random_state' in best_params.keys():
                    best_params['random_state'] = 42

                model = utils.Model(model_class(**best_params),scaler,imputer if config['fill_na'] != 0 else None,None,None)
                
                best_features = utils.rfe(utils.Model(model_class(**best_params),scaler,imputer if config['fill_na'] != 0 else None,None,None),X_train,y_train.values if isinstance(y_train,pd.Series) else y_train,IDs_train,CV,scoring,problem_type,cmatrix=cmatrix,priors=None,threshold=threshold,round_values=config['round_values'],covariates=covariates if regress_out else None,fill_na=config['fill_na'],regress_out_method=config['regress_out_method'])[0] if feature_selection else X_train.columns
                
                model.train(X_train[best_features],y_train.values if isinstance(y_train,pd.Series) else y_train,covariates if regress_out else None, config['fill_na'],config['regress_out_method'])

                shap_values = utils.run_shap_analysis(model,X_train[best_features],y_train,IDs_train,CV,fill_na=config['fill_na'])

                feature_importance_file = f'shap_feature_importance_{model_type}_{task}_{dimension}_{y_label}_{model_type}_shuffled_calibrated.csv'.replace('__','_')

                if not shuffle_labels:
                    feature_importance_file = feature_importance_file.replace('_shuffled','')
                if not calibrate:
                    feature_importance_file = feature_importance_file.replace('_calibrated','')

                feature_importance_fig_file = feature_importance_file.replace('.csv','.png')

                Path(results_dir).mkdir(parents=True, exist_ok=True)
                
                Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed).mkdir(parents=True,exist_ok=True)
                Path(results_dir,f'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed).mkdir(parents=True,exist_ok=True)
                try:
                    mean_shap_values = shap_values.mean()
                    mean_shap_values.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,feature_importance_file))
                    # Summary Plot (Beeswarm)
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values.values.astype(float), X_train[best_features], feature_names=best_features, show=False)
                    plt.title(f'SHAP Summary', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(Path(results_dir, feature_importance_fig_file), dpi=300)
                    plt.close()
                except:

                    if hasattr(model.model,'coef_'):
                        feature_importance = np.abs(model.model.coef_[0])
                        coef = pd.DataFrame({'feature':best_features,'importance':feature_importance / np.sum(feature_importance)}).sort_values('importance',ascending=False)
                        coef.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,feature_importance_file),index=False)
                    elif hasattr(model.model,'feature_importance'):
                        feature_importance = model.model.feature_importance
                        feature_importance = pd.DataFrame({'feature':best_features,'importance':feature_importance}).sort_values('importance',ascending=False)
                        feature_importance.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,feature_importance_file),index=False)
                    elif hasattr(model.model,'get_booster'):
                        feature_importance = pd.DataFrame({'feature':best_features,'importance':model.model.feature_importances_}).sort_values('importance',ascending=False)
                        feature_importance.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,feature_importance_file),index=False)
                    elif hasattr(model.model,'feature_importances_'):
                        feature_importance = model.model.feature_importances_
                        feature_importance = pd.DataFrame({'feature':best_features,'importance':feature_importance}).sort_values('importance',ascending=False)
                        feature_importance.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,feature_importance_file),index=False)
                    
                    else:
                        print(task,dimension,f'No feature importance available for {model_type}')
                    
                pickle.dump(model.model,open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,f'model_{model_type}.npy'),'wb'))
                pickle.dump(model.scaler,open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,f'scaler_{model_type}.npy'),'wb'))
                pickle.dump(model.imputer,open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed,f'imputer_{model_type}.npy'),'wb'))  
                
        if problem_type == 'reg':
            best_models = pd.concat((best_models,corr_results), axis=1) 

        best_models = best_models.sort_values(by=['y_label',f'{scoring}_extremo'],ascending=[True,False])
        
        if problem_type == 'reg':
            p_vals = best_models['p_value'].values
            reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method=correction)
            best_models['p_value_corrected'] = p_vals_corrected
            best_models['p_value_corrected'] = best_models['p_value_corrected'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
            best_models['p_value'] = best_models['p_value'].apply(lambda x: f'{x:.2e}' if x < 0.001 else f'{x:.3f}')
            best_models['r'] = best_models['r'].apply(lambda x: f'{x:.3f}' if not pd.isna(x) else np.nan)
            best_models['correction_method'] = correction

            filename = filename.replace('.csv',f'_corr_{covars[-1]}.csv') if len(covars) != 0 else filename.replace('.csv','_corr.csv')
            best_models.to_csv(Path(results_dir,filename),index=False)
