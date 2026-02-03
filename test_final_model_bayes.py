import pandas as pd
import numpy as np
from pathlib import Path
import pickle, sys, warnings, json, os
from scipy.stats import pearsonr, spearmanr, shapiro

warnings.filterwarnings('ignore')

from scipy.stats import bootstrap

from matplotlib import pyplot as plt
import seaborn as sns

from expected_cost.ec import CostMatrix

from statsmodels.stats.multitest import multipletests
from pingouin import partial_corr

import utils

late_fusion = False

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
id_col = config['id_col']
scaler_name = config['scaler_name']
stat_folder = config['stat_folder']
kfold_folder = config['kfold_folder']
hyp_opt = config["n_iter"] > 0 
shuffle_labels = config['shuffle_labels']
feature_selection = config['feature_selection']
filter_outliers = config['filter_outliers']
round_values = config['round_values']
cut_values = config['cut_values'] > 0
n_boot_test = int(config['n_boot_test'])
n_boot_train = int(config['n_boot_train'])
calibrate = bool(config["calibrate"])
overwrite = bool(config["overwrite"])
scoring = config['scoring_metric']
problem_type = config['problem_type']
y_labels = config['y_labels']

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

try:
    thresholds = main_config['thresholds'][project_name]
except KeyError:
    thresholds =[None]

data_file = main_config["data_file"][project_name]

try:
    covars = main_config["covars"][project_name] if problem_type == 'reg' else []
except:
    covars = []
    
metrics_names_ = main_config['metrics_names'][problem_type]

try:
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
except:
    cmatrix = None

##---------------------------------PARAMETERS---------------------------------##
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
save_dir = Path(str(data_dir).replace('data','results'))    

results_test = pd.DataFrame()

correction = 'fdr_bh'

method = 'pearson'

filename = f'best_best_models_{scoring}_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_filter_outliers_round_cut_shuffled_calibrated_bayes_{config["version"]}_corr_{covars[-1]}.csv'.replace('__','_') if len(covars) > 0 else f'best_best_models_{scoring}_{kfold_folder}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_filter_outliers_round_cut_shuffled_calibrated_bayes_{config["version"]}.csv'.replace('__','_')

if not hyp_opt:
    filename = filename.replace("_hyp_opt","")
if not feature_selection:
    filename = filename.replace("_feature_selection","")
if not filter_outliers:
    filename = filename.replace("_filter_outliers","")
if not round_values:
    filename = filename.replace("_round","")
if not cut_values:
    filename = filename.replace("_cut","")
if not shuffle_labels:
    filename = filename.replace("_shuffled","")
if not calibrate:
    filename = filename.replace("_calibrated","")
if not problem_type == 'reg':
    filename = filename.replace(f"_corr_{covars[-1]}","") if len(covars) > 0 else filename.replace("_corr_","")

best_models = pd.read_csv(Path(results_dir,filename))

for r, row in best_models.iterrows():
    task = row['task']
    dimension = row['dimension']
    model_type = row['model_type']
    random_seed_test = row['random_seed_test']
    if str(random_seed_test) == 'nan':
        random_seed_test = ''
    y_label = row['y_label']

    print(task,dimension,model_type,y_label)
    try:
        trained_model = np.load(open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed_test,f'model_{model_type}.npy'),'rb'),allow_pickle=True)
        trained_scaler = np.load(open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed_test,f'scaler_{model_type}.npy'),'rb'),allow_pickle=True)
        trained_imputer = np.load(open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed_test,f'imputer_{model_type}.npy'),'rb'),allow_pickle=True)
    except:
        continue

    path_to_results = Path(save_dir,task,dimension,kfold_folder,y_label,stat_folder,scoring,'hyp_opt','feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '',"shuffle" if shuffle_labels else "",config['version'])
    
    data_train = pd.read_csv(Path(path_to_results,random_seed_test,'data_train.csv'))
    columns_train = list(set(data_train.columns)  - set(y_labels + ['id']))

    X_train = np.load(open(Path(path_to_results,random_seed_test,'X_train.npy'),'rb'),allow_pickle=True)
    y_train = np.load(open(Path(path_to_results,random_seed_test,'y_train.npy'),'rb'),allow_pickle=True)
    IDs_train = np.load(open(Path(path_to_results,random_seed_test,'IDs_train.npy'),'rb'),allow_pickle=True)
    
    X_test = pd.read_csv(Path(path_to_results,random_seed_test,'data_test.csv'))
    columns_test = list(set(data_train.columns)  - set(y_labels + ['id']))

    columns = list(set(columns_train) & set(columns_test))

    y_test = np.load(open(Path(path_to_results,random_seed_test,'y_test.npy'),'rb'),allow_pickle=True)
    IDs_test = np.load(open(Path(path_to_results,random_seed_test,'IDs_test.npy'),'rb'),allow_pickle=True)
    params = trained_model.get_params()

    features = list(set(trained_model.feature_names_in_).intersection(set(columns_test)))

    if not isinstance(X_train,pd.DataFrame):
        X_train = pd.DataFrame(data=X_train,columns=columns_train)    
    
    if not isinstance(X_test,pd.DataFrame):
        X_test = pd.DataFrame(data=X_test,columns=columns_test)

    if 'probability' in params.keys():
        params['probability'] = True
        
    regress_out = list(set(json.load(open(Path(path_to_results,random_seed_test,'config.json'),'rb'))['regress_out']) - set(['']))

    regress_out_method = json.load(open(Path(path_to_results,random_seed_test,'config.json'),'rb'))['regress_out_method']
    fill_na = json.load(open(Path(path_to_results,random_seed_test,'config.json'),'rb'))['fill_na']
    
    covariates_regress_out = pd.read_csv(Path(data_dir,data_file))[[id_col]+regress_out] if len(regress_out) > 0 else None

    covariates_regress_out_train = covariates_regress_out[covariates_regress_out[id_col].isin(np.unique(IDs_train))].reset_index(drop=True) if covariates_regress_out is not None else None
    covariates_regress_out_train.drop(id_col,axis=1,inplace=True) if covariates_regress_out is not None else None
    
    covariates_regress_out_test = pd.read_csv(Path(path_to_results,random_seed_test,'data_test.csv'))[[id_col]+regress_out] if len(regress_out) > 0 else None
    covariates_regress_out_test.drop(id_col,axis=1,inplace=True) if covariates_regress_out is not None else None

    metrics_names = list(set(metrics_names_) - set(['roc_auc','f1','recall','precision'])) if cmatrix is not None or len(np.unique(y_train)) > 2 else metrics_names_
    
    model = utils.Model(type(trained_model)(**params),type(trained_scaler),type(trained_imputer) if config['fill_na'] != 0 else None)

    covars = config['covariates'] if problem_type == 'reg' else []
        
    covariates = pd.read_csv(Path(data_dir,data_file))[[config["id_col"]]+covars]

    covariates[id_col] = covariates[id_col].astype(str)

    model.train(X_train[features],y_train.values if isinstance(y_train,pd.Series) else y_train,covariates=covariates_regress_out_train if regress_out else None,fill_na=config['fill_na'])

    try:
        outputs = model.eval(X_test[features],problem_type,covariates=covariates_regress_out_test if regress_out else None,fill_na=config['fill_na'])
    except:
        continue

    subfolders = [
            task, dimension,
            config['kfold_folder'], y_label, config['stat_folder'],scoring,
            'hyp_opt' if config['n_iter'] > 0 else '','feature_selection' if config['feature_selection'] else '',
            'filter_outliers' if config['filter_outliers'] and problem_type == 'reg' else '',
            'shuffle' if config['shuffle_labels'] else '',random_seed_test,config['version']
        ]

    path_to_save = results_dir.joinpath(*[str(s) for s in subfolders if s])
    
    np.save(open(str(Path(path_to_save,f'outputs_test_{model_type}.npy')),'wb'),outputs)
    # Prepare data for bootstrap: a tuple of index arrays to resample
    data_indices = (np.arange(y_test.shape[-1]),)

    # Define the statistic function with data baked in
    stat_func = lambda indices: utils._calculate_metrics(
        indices, outputs, y_test.values if isinstance(y_test,pd.Series) else y_test,
        metrics_names, problem_type, cmatrix
    )

    # 1. Calculate the point estimate (the actual difference on the full dataset)
    point_estimates = stat_func(data_indices[0])

    # 2. Calculate the bootstrap confidence interval
    try:
        # Try the more accurate BCa method first
        res = bootstrap(
            data_indices,
            stat_func,
            n_resamples=n_boot_test, # Use configured n_boot
            method=config["bootstrap_method"],
            vectorized=False,
            random_state=42
        )
        bootstrap_method = config["bootstrap_method"]

    except ValueError as e:
        # If BCa fails (e.g., due to degenerate samples), fall back to percentile
        print(f"WARNING: {config['bootstrap_method']} method failed for {task}/{dimension}/{y_label}. Falling back to 'percentile'. Error: {e}")
        res = bootstrap(
            data_indices,
            stat_func,
            n_resamples=n_boot_test,
            method='percentile',
            vectorized=False,
            random_state=42
        )
        bootstrap_method = 'percentile'
    
    best_models.loc[r,'bootstrap_method_holdout'] = bootstrap_method
    for i, metric in enumerate(metrics_names):
        est = point_estimates[i]
        ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
        best_models.loc[r,f'{metric}_holdout'] = f"{est:.3f}, ({ci_low:.3f}, {ci_high:.3f})"
    
    if problem_type == 'reg':
        predictions_test = pd.DataFrame({'id':IDs_test.flatten(),'y_pred':outputs.flatten(),'y_true':y_test.flatten()})
    else:
        _, y_pred = utils.get_metrics_clf(outputs, y_test, [], cmatrix=cmatrix, priors=None, threshold=None)
        predictions_test = {'id':IDs_test.values.flatten() if isinstance(IDs_test,pd.Series) else IDs_test,'y_pred':y_pred.flatten(),'y_true':y_test.values.flatten() if isinstance(y_test,pd.Series) else y_test.flatten()}
        for c in range(outputs.shape[-1]):
            predictions_test[f'outputs_class_{c}'] = outputs[:,c].flatten()
        predictions_test = pd.DataFrame(predictions_test)

    predictions_test = predictions_test.drop_duplicates('id')

    if problem_type == 'reg':
        try:
            idx = best_models[(best_models['task'] == task) &
                                (best_models['dimension'] == dimension) &
                                (best_models['y_label'] == y_label) &
                                (best_models['model_type'] == model_type)].index[0]

        except:
            continue
        
        sns.set_theme(style="whitegrid")  # Fondo blanco con grid sutil
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.titlesize": 26,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20
        })
        covars = ['age','education','sex'] if '_res_' not in y_label else []
        if 'data_file_test' in main_config.keys():
            data_file_test = main_config['data_file_test'][project_name]
            covariates_test = pd.read_csv(Path(data_dir,data_file_test))[[id_col]+covars]
        else:
            covariates_test = pd.read_csv(Path(data_dir,data_file))[[id_col]+covars]

        covariates_test[id_col] = covariates_test[id_col].astype(str)
        predictions_test[id_col] = predictions_test[id_col].astype(str)

        if not covariates.empty:
            predictions_test = pd.merge(predictions_test,covariates_test,on=id_col,how='inner')

        Path(results_dir,f'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'], random_seed_test).mkdir(exist_ok=True,parents=True)

        with open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed_test,f'predictions_test.npy'),'wb') as f:
            pickle.dump(predictions_test,f)
        predictions_test.to_csv(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],random_seed_test,f'predictions_test.csv'),index=False)

        try:
            results = partial_corr(data=predictions_test,x='y_pred',y='y_true',covar=covars,method=method)
            n, r, ci, p = results.loc[method,'n'], results.loc[method,'r'], results.loc[method,'CI95%'], results.loc[method,'p-val']
        except:
            r, p = pearsonr(predictions_test['y_pred'], predictions_test['y_true']) if method == 'pearson' else spearmanr(predictions_test['y_true'], predictions_test['y_pred'])
            n = predictions_test.shape[0]
            ci = np.nan

        best_models.loc[idx,['r_holdout','p_value_corrected_holdout','p_holdout','method','n_holdout','95_ci_holdout','covars_holdout','correction_method_holdout']] = [r,np.nan,p,method,n,str(ci),str(covars),np.nan]

        save_path = Path(results_dir, f'plots', task, dimension, y_label,
                        stat_folder,config["bootstrap_method"],'bayes',scoring,
                        'hyp_opt' if hyp_opt else '',
                        'filter_outliers' if filter_outliers else '','rounded' if round_values else '', 'cut' if cut_values else '','shuffle' if shuffle_labels else '',config['version'],
                        f'{model_type}_test.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.regplot(
            x='y_pred', y='y_true', data=predictions_test,
            scatter_kws={'alpha': 0.6, 's': 50, 'color': '#c9a400'},  # color base
            line_kws={'color': 'black', 'linewidth': 2}
        )

        plt.xlabel('Predicted Value')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('True Value')

        plt.text(0.05, 0.95,
                f'$r$ = {r:.2f}\n$p$ = {np.round(p,3) if p >= .001 else "< .001"}',
                fontsize=30,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        tasks_dict = {'animales':'Semantic',
        "grandmean": "Average",
        "nps":"Cognition",
        "brain":"Brain"}

        plt.title(f'{task} | {y_label}', fontsize=25, pad=15)

        plt.tight_layout()
        plt.grid(False)
        try:
            plt.savefig(save_path, dpi=300)
        except:
            continue
        plt.close()

if problem_type == 'reg':
    p_vals = best_models['p_holdout'].values
    reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method=correction)
    best_models['p_value_corrected_holdout'] = p_vals_corrected
    best_models['p_value_corrected_holdout'] = best_models['p_value_corrected_holdout'].apply(lambda x: f"{x:.2e}" if x < 0.01 else f"{x:.3f}")
    best_models['r_holdout'] = best_models['r_holdout'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else np.nan)
    best_models['p_holdout'] = best_models['p_holdout'].apply(lambda x: f"{x:.2e}" if x < 0.01 else f"{x:.3f}")

    best_models['correction_method_holdout'] = correction

best_models.to_csv(Path(results_dir,filename.replace('.csv','_test.csv')),index=False)