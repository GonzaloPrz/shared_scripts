import pandas as pd
import numpy as np
from pathlib import Path
import itertools, pickle, sys, warnings, json, os
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNNR
from xgboost import XGBRegressor as xgboostr

from scipy.stats import bootstrap

from sklearn.neighbors import KNeighborsRegressor

from sklearn.utils import resample 

from expected_cost.ec import *

from expected_cost.calibration import calibration_train_on_heldout
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal

import utils

late_fusion = False

##---------------------------------PARAMETERS---------------------------------##
config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
calibrate = config["calibrate"]
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = bool(config['n_iter'])
feature_selection = bool(config['n_iter_features'] > 0)
filter_outliers = config['filter_outliers']
n_models_ = int(config["n_models"])
early_fusion = bool(config["early_fusion"])
bayesian = bool(config["bayesian"])
n_boot_test = int(config["n_boot_test"])
n_boot_train = int(config["n_boot_train"])
calibrate = bool(config["calibrate"])
overwrite = bool(config["overwrite"])
regress_out = config["regress_out"]

if calibrate:    
    calmethod = AffineCalLogLoss
    calparams = {'bias':True, 'priors':None}
else:
    calmethod = None
    calparams = None

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = config['tasks']
test_size = main_config['test_size'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]

scoring_metrics = config['scoring_metrics']
problem_type = config['problem_type']
if problem_type == 'clf':
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
else:
    cmatrix = None
    
parallel = bool(config["parallel"])

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

##---------------------------------PARAMETERS---------------------------------##
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
save_dir = Path(str(data_dir).replace('data','results'))    

if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None
imputer = KNNImputer

models_dict = {'clf':{'lr': LogisticRegression,
                    'svc': SVC, 
                    'xgb': XGBClassifier,
                    'knnc': KNNC,
                    'lda': LDA,
                    'qda':QDA,
                    'nb': GaussianNB},
                
                'reg':{'lasso':Lasso,
                        'ridge':Ridge,
                        'elastic':ElasticNet,
                        'knnr':KNNR,
                        'svr':SVR,
                        'xgb':xgboostr
                    }
}

for task,scoring in itertools.product(tasks,scoring_metrics):
    
    extremo = 1 if any(x in scoring for x in ['norm','error']) else 0
    ascending = True if extremo == 1 else False

    dimensions = [folder.name for folder in Path(save_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        for y_label in y_labels:
            print(y_label)
            path_to_results = Path(save_dir,task,dimension,kfold_folder, y_label,stat_folder,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '',"regress_out" if regress_out else "","shuffle" if shuffle_labels else "")

            if not path_to_results.exists():
                continue
            
            random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir() if 'random_seed' in folder.name]
            random_seeds_test.append('')
            
            for random_seed_test in random_seeds_test:
                if int(config["n_models"] == 0):
                    files = [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['all_models_','dev',config["bootstrap_method"],'calibrated'])] if calibrate else [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['all_models_','dev',config["bootstrap_method"]]) and 'calibrated' not in file.stem]
                else:
                    files = [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['best_models_','dev',config["bootstrap_method"],'calibrated'])] if calibrate else [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['best_models_','dev',config["bootstrap_method"]]) and 'calibrated' not in file.stem]
                
                if len(files) == 0:
                    continue

                X_dev = pickle.load(open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'rb'))[0,0]
                y_dev = pickle.load(open(Path(path_to_results,random_seed_test,'y_dev.pkl'),'rb'))[0,0]
                IDs_dev = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_dev.pkl'),'rb'))[0,0]
                X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))   
                y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
                IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))
                
                if X_test.shape[0] == 0:
                    continue
                
                for file in files:
                    model_name = file.stem.split('_')[2]

                    if file.suffix != '.csv':
                        continue

                    filename_to_save = file.name.replace('dev','test')
                    print(model_name)
                    
                    results_dev = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)

                    n_models = int(n_models_*results_dev.shape[0]) if n_models_ else results_dev.shape[0]
                    all_features = [col for col in results_dev.columns if any(f'{x}__{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__'))) or col =='group']
                    
                    if not isinstance(X_dev,pd.DataFrame):
                        X_dev = pd.DataFrame(columns=all_features,data=X_dev)
                    
                    if not isinstance(X_test,pd.DataFrame):
                        X_test = pd.DataFrame(columns=all_features,data=X_test)
                    metrics_names = main_config["metrics_names"][problem_type] if len(np.unique(y_dev)) == 2 else list(set(main_config["metrics_names"][problem_type]) - set(['roc_auc','f1','recall']))

                    if Path(file.parent,filename_to_save).exists() and overwrite == False:
                        print(f"Testing already done")
                        continue
                            
                    def parallel_process(index):
                        
                        features = [col for col in all_features if results_dev.loc[index,col]== 1]
                        params = [col for col in results_dev.columns if all(x not in col for x in  all_features + metrics_names + [y_label,config["id_col"],'Unnamed: 0','threshold','index',f'{scoring}_extremo','bootstrap_method'])]

                        params_dict = {param:results_dev.loc[index,param] for param in params if str(results_dev.loc[index,param]) != 'nan'}

                        if 'gamma' in params_dict.keys():
                            try: 
                                params_dict['gamma'] = float(params_dict['gamma'])
                            except:
                                pass

                        if 'random_state' in params_dict.keys():
                            params_dict['random_state'] = int(params_dict['random_state'])
                        
                        outputs = utils.test_model(models_dict[problem_type][model_name],params_dict,scaler,imputer, X_dev[features], y_dev, X_test[features], problem_type=problem_type)
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
                            print(f"WARNING: {config['bootstrap_method']} method failed for {tasks}/{dimensions}/{y_label}. Falling back to 'percentile'. Error: {e}")
                            res = bootstrap(
                                data_indices,
                                stat_func,
                                n_resamples=n_boot_test,
                                method='percentile',
                                vectorized=False,
                                random_state=42
                            )
                            bootstrap_method = 'percentile'

                        result_row = dict(results_dev.iloc[index])
                        result_row.update({'bootstrap_method':bootstrap_method})
                        for i, metric in enumerate(metrics_names):
                            est = point_estimates[i]
                            ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
                            result_row.update({f'{metric}_holdout': f"{est:.5f}, ({ci_low:.5f}, {ci_high:.5f})"})
                        
                        return result_row, outputs
                    
                    
                    parallel_results = Parallel(n_jobs=-1 if parallel else 1)(delayed(parallel_process)(index) for index in np.arange(n_models))
                    
                    all_results = pd.concat((pd.DataFrame(result[0],index=[0]) for result in parallel_results),ignore_index=True)
                    
                    all_results.to_csv(Path(path_to_results,random_seed_test,filename_to_save))
                    
                    outputs_test = np.stack([result[1] for result in parallel_results],axis=0)
                    
                    outputs_filename = f'cal_outputs_test_{model_name}.pkl' if calibrate else f'outputs_test_{model_name}.pkl'
                    with open(Path(file.parent,outputs_filename),'wb') as f:
                        pickle.dump(outputs_test,f)