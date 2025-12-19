import numpy as np
import pandas as pd
from pathlib import Path
import math 
import logging, sys
import torch

from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit, StratifiedShuffleSplit, LeaveOneOut
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgboost
from xgboost import XGBRegressor as xgboostr
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys, json
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 
import warnings,argparse,os,multiprocessing
from psrcal.calibration import AffineCalLogLoss

from expected_cost.ec import *
from expected_cost.utils import *

warnings.filterwarnings('ignore')

from random import randint as randint_random 

import utils

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train models with hyperparameter optimization and feature selection'
    )
    parser.add_argument('--project_name', default='arequipa',type=str,help='Project name')
    parser.add_argument('--stats', type=str, default='count_mean_ratio', help='Stats to be considered (default = all)')
    parser.add_argument('--shuffle_labels', type=int, default=0, help='Shuffle labels flag (1 or 0)')
    parser.add_argument('--stratify', type=int, default=1, help='Stratification flag (1 or 0)')
    parser.add_argument('--calibrate', type=int, default=0, help='Whether to calibrate models')
    parser.add_argument('--n_folds', type=float, default=5, help='Number of folds for cross validation')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of hyperparameter iterations')
    parser.add_argument('--n_iter_features', type=int, default=20, help='Number of feature sets to try and select from')
    parser.add_argument('--feature_sample_ratio', type=float, default=0.5, help='Feature-to-sample ratio: number of features in each feature set = ratio * number of samples in the training set')
    parser.add_argument('--n_seeds_train',type=int,default=10,help='Number of seeds for cross-validation training')
    parser.add_argument('--n_seeds_shuffle',type=int,default=10,help='Number of seeds for shuffling')
    parser.add_argument('--scaler_name', type=str, default='StandardScaler', help='Scaler name')
    parser.add_argument('--id_col', type=str, default='id', help='ID column name')
    parser.add_argument('--n_models',type=float,default=0,help='Number of hyperparameter combinatios to try and select from  to train')
    parser.add_argument('--n_boot',type=int,default=1000,help='')
    parser.add_argument('--bayesian',type=int,default=0,help='Whether to calculate bayesian credible intervals or bootstrap confidence intervals')
    parser.add_argument('--shuffle_all',type=int,default=0,help='Whether to shuffle all models or only the best ones')
    parser.add_argument('--filter_outliers',type=int,default=0,help='Whether to filter outliers in regression problems')
    parser.add_argument('--early_fusion',type=int,default=0,help='Whether to perform early fusion')
    parser.add_argument('--n_boot_test',type=int,default=1000,help='Number of bootstrap samples for holdout')
    parser.add_argument('--n_boot_train',type=int,default=0,help='Number of bootstrap samples of training samples while performing model testing')
    parser.add_argument('--overwrite',type=int,default=0,help='Whether to overwrite past results or not')
    parser.add_argument('--parallel',type=int,default=1,help='Whether to parallelize processes or not')
    parser.add_argument('--n_seeds_test',type=int,default=1,help='Number of seeds for testing')
    parser.add_argument('--bootstrap_method',type=str,default='bca',help='Bootstrap method [bca, percentile, basic]')
    parser.add_argument('--add_dem',type=int,default=0,help='Whether to add demographics as covariates')
    parser.add_argument('--regress_out',type=int,default=0,help='Whether to regress out covariates from target variable before training')

    return parser.parse_args()

def load_configuration(args):
    # Global configuration dictionaries
    config = dict(
        project_name = args.project_name,
        stats = str(args.stats),
        shuffle_labels = bool(args.shuffle_labels),
        shuffle_all = bool(args.shuffle_all),
        stratify = bool(args.stratify),
        calibrate = bool(args.calibrate),
        n_folds = float(args.n_folds),
        n_iter = float(args.n_iter),
        n_iter_features = float(args.n_iter_features),
        feature_sample_ratio = args.feature_sample_ratio,
        n_seeds_train = float(args.n_seeds_train) if args.n_folds != -1 else float(1),
        n_seeds_shuffle = float(args.n_seeds_shuffle) if args.shuffle_labels else float(0),
        scaler_name = args.scaler_name,
        id_col = args.id_col,
        n_models = float(args.n_models),
        n_boot = float(args.n_boot),
        bayesian = bool(args.bayesian),
        filter_outliers = bool(args.filter_outliers),
        early_fusion = bool(args.early_fusion),
        n_boot_test = float(args.n_boot_test),
        n_boot_train = float(args.n_boot_train),
        overwrite = bool(args.overwrite),
        parallel = bool(args.parallel),
        n_seeds_test = float(args.n_seeds_test) if args.n_folds != -1 else float(0),
        bootstrap_method = args.bootstrap_method,
        bayes = False,
        add_dem = bool(args.add_dem),
        regress_out = bool(args.regress_out)
    )

    return config

args = parse_args()
config = load_configuration(args)
project_name = config['project_name']

logging.info('Configuration loaded. Starting training...')
logging.info('Training completed.')

##------------------ Configuration and Parameter Parsing ------------------##
multiprocessing.set_start_method('spawn', force=True)

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    data_dir = home / 'data' / project_name
else:
    data_dir = Path('D:/CNC_Audio/gonza/data', project_name)

results_dir = Path(str(data_dir).replace('data', 'results'))

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
problem_type = 'reg' if any(x in project_name.lower() for x in ['ceac','reg']) else 'clf'
scoring_metrics = 'roc_auc' if problem_type == 'clf' else 'r2'

config['test_size'] = float(test_size)
config['data_file'] = data_file
config['tasks'] = tasks
config['single_dimensions'] = single_dimensions        
config['scoring_metrics'] = scoring_metrics
config['problem_type'] = problem_type
config['y_labels'] = y_labels
config['avoid_stats'] = list(set(['min','max','median','skewness','kurtosis','std','mean', 'variability','count','ratio']) - set(config['stats'].split('_'))) if config['stats'] != '' else []
config['stat_folder'] = '_'.join(sorted(config['stats'].split('_')))

config['random_seeds_train'] = [float(3**x) for x in np.arange(1, config['n_seeds_train']+1)]
config['random_seeds_test'] = [float(3**x) for x in np.arange(1, config['n_seeds_test']+1)] if config['test_size'] > 0 else ['']
config['random_seeds_shuffle'] = [float(3**x) for x in np.arange(1, config['n_seeds_shuffle']+1)] if config['shuffle_labels'] else ['']

n_folds = config['n_folds']

if config['calibrate']:
    calmethod = AffineCalLogLoss
    calparams = {'bias':True, 'priors':None}
else:
    calmethod = None
    calparams = None

models_dict = {
        'clf': {
            'lr': LR,
            #'svc': SVC,
            'knnc': KNNC,
            'xgb': xgboost,
            #'lda': LDA,
            'qda': QDA,
            #'nb':GaussianNB
        },
        'reg': {
            'lasso': Lasso,
            'ridge': Ridge,
            'elastic': ElasticNet,
            'svr': SVR,
            'xgb': xgboostr,
            'knnr': KNNR
        }
    }

with Path(Path(__file__).parent,'default_hp.json').open('rb') as f:
    default_hp = json.load(f)

hp_ranges = {
        'lr': {'C': [x*10**y for x,y in itertools.product(range(1,9),range(-3, 2))]},
        'svc': {'C': [x*10**y for x,y in itertools.product(range(1,9),range(-3, 2))], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'probability': [True]},
        'xgb': {'n_estimators': [x*10**y for x,y in itertools.product(range(1,6),range(1,3))], 'max_depth': [1, 2, 3, 4], 'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'lda': {'solver': ['svd', 'lsqr', 'eigen']},
        'qda': {'reg_param': [x*10**y for x,y in itertools.product(range(0,5),range(-4, -1))],
                  'tol': [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))]},
        'nb': {'priors':[None]},
        'ridge': {'alpha': [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  'tol': [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  'solver': ['auto'],
                  'max_iter': [5000],
                  'random_state': [42]},
        'lasso': {'alpha': [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  'tol': [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  'max_iter': [5000], 
                  'random_state': [42]},
        'elastic': {'alpha': [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                    'l1_ratio': [x**-1 for x in range(1,10)], 
                    'tol': [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  'max_iter': [5000],
                  'random_state': [42]},
        'svr': {'C': [x*10**y for x,y in itertools.product(range(1,9),range(-3, 2))], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto'],
}
}
##------------------ Main Model Training Loop ------------------##

for y_label, task in itertools.product(y_labels, tasks):
    print(y_label)
    print(task)
    # Determine feature dimensions. For projects with a dictionary, pick based on the task.
    dimensions = []
    if isinstance(single_dimensions,dict):
        single_dims = single_dimensions[task]
    else:
        single_dims = single_dimensions
    
    if isinstance(single_dims, list) and config['early_fusion']:
        for ndim in range(1, len(single_dims)+1):
            for dimension in itertools.combinations(single_dims, ndim):
                dimensions.append('__'.join(dimension))
    else:
        dimensions = single_dims
    
    for dimension in dimensions:
        print(dimension)
        logging.info(f'Processing: y_label={y_label}, task={task}, dimension={dimension}')
        # Load dataset. Use CSV or Excel based on file extension.
        data_file = data_file
        data_path = data_dir / (data_file if data_file.endswith('.csv') else data_file)
        if problem_type == 'clf':
            data = pd.read_csv(data_path if data_file.endswith('.csv') else data_path.with_suffix('.csv'))
        else:
            # For regression: Excel if available; default to CSV.
            data = pd.read_excel(data_path) if data_path.suffix in ['.xlsx', '.xls'] else pd.read_csv(data_path)

        data.dropna(axis=1,how='all',inplace=True)

        # Identify feature columns (avoid stats and other unwanted columns)
        features = [col for col in data.columns if any(f'{x}__{y}__' in col 
                    for x,y in itertools.product(task.split('__'), dimension.split('__'))) 
                    and not isinstance(data.iloc[0][col], str) 
                    and all(f'_{x}' not in col for x in config['avoid_stats'] + ['query', 'timestamp'])]

        if len(features) == 0:
            continue
        
        if "group_as_feature" in config["project_name"] and y_label != "group":
            features = features + ["group"]

        if config['add_dem']:
            for col in set(['sex','age','education','handedness']).intersection(set(data.columns)):
                
                data[f'{task}__dem__{col}'] = data[col]

            demographic_features = [f'{task}__dem__{col}' for col in data.columns if col in ['sex','age','education','handedness']]

            features.extend(demographic_features)
            dimension = dimension + '__dem' if dimension != '' else 'dem'
        
        # Select only the desired features along with the target and id
        data = data[features + [y_label, config['id_col']]]
        data = data.dropna(subset=[y_label])
                
        if any(data[features].isna().sum()/data.shape[0] > .2):
            data = data.dropna(subset=features) 

        # For regression, optionally filter outliers
        if problem_type == 'reg' and config['filter_outliers']:
            data = data[np.abs(data[y_label]-data[y_label].mean()) <= (3*data[y_label].std())]
        
        # Separate features, target and ID.
        ID = data.pop(config['id_col'])
        y = data.pop(y_label)
        strat_col = y if (config['stratify'] and problem_type == 'clf') else None

        # Iterate over each model defined for this problem type.
        for model_key, model_class in models_dict[problem_type].items():
            # Determine held-out settings based on hyperparameter or feature iterations.
            held_out = (config['n_iter'] > 0 or config['n_iter_features'] > 0)
            if held_out:
                n_samples_dev = int(data.shape[0] * (1 - config['test_size']))
            else:
                n_samples_dev = data.shape[0]
                n_seeds_test = 0
                
            random_seeds_test = config['random_seeds_test']

            if n_folds == 0:
                n_folds = int(n_samples_dev / np.unique(y).shape[0])
                CV = (StratifiedKFold(n_splits=n_folds, shuffle=True)
                            if strat_col is not None
                            else KFold(n_splits=n_folds, shuffle=True))
                config["kfold_folder"] = f'l{np.unique(y).shape[0]}ocv'
                n_max = n_samples_dev - np.unique(y).shape[0]
            elif n_folds == -1:
                CV = LeaveOneOut()
                config["kfold_folder"] = 'loocv'
                n_max = n_samples_dev - 1
            elif n_folds < 1:
                CV = (StratifiedShuffleSplit(n_splits=1,test_size=n_folds)
                            if strat_col is not None
                            else ShuffleSplit(n_splits=1,test_size=n_folds))
                n_max = int(n_samples_dev*(1-n_folds))
                config["kfold_folder"] = f'{int(n_folds*100)}pct'
            else: 
                n_folds = int(n_folds)
                CV = (StratifiedKFold(n_splits=n_folds, shuffle=True)
                            if strat_col is not None
                            else KFold(n_splits=n_folds, shuffle=True))
                n_max = int(n_samples_dev*(1-1/n_folds))
                config["kfold_folder"] = f'{n_folds}_folds'
                
            # Construct a path to save results (with clear folder names)
            subfolders = [
                task, dimension,
                config['kfold_folder'], y_label, config['stat_folder'],
                'hyp_opt' if config['n_iter'] > 0 else '',
                'feature_selection' if config['n_iter_features'] > 0 else '',
                'filter_outliers' if config['filter_outliers'] and problem_type == 'reg' else '','regress_out' if config['regress_out'] else '',
                'shuffle' if config['shuffle_labels'] else ''
            ]
            path_to_save = results_dir.joinpath(*[str(s) for s in subfolders if s])
            path_to_save.mkdir(parents=True, exist_ok=True)
            
            for random_seed_test in random_seeds_test:
                Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '').mkdir(exist_ok=True,parents=True)
                X_dev, y_dev, IDs_dev, outputs, X_test, y_test, IDs_test = np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0)

                if test_size > 0:
                    X_train_, X_test_, y_train_, y_test_, ID_train_, ID_test_ = train_test_split(
                            data, y, ID,
                            test_size=config['test_size'],
                            random_state=int(random_seed_test),
                            shuffle=True,
                            stratify=strat_col)
                    
                    # Reset indexes after split.
                    X_train_.reset_index(drop=True, inplace=True)
                    X_test_.reset_index(drop=True, inplace=True)
                    y_train_ = y_train_.reset_index(drop=True)
                    y_test_ = y_test_.reset_index(drop=True)
                    ID_train_ = ID_train_.reset_index(drop=True)
                    ID_test_ = ID_test_.reset_index(drop=True)
                else:
                    X_train_, y_train_, ID_train_ = data.reset_index(drop=True), y.reset_index(drop=True), ID.reset_index(drop=True)
                    X_test_, y_test_, ID_test_ = pd.DataFrame(), pd.Series(), pd.Series()
                
                hp_ranges.update({'knnc':{'n_neighbors': np.arange(1,n_max)}})
                hp_ranges.update({'knnr':{'n_neighbors': np.arange(1,n_max)}})
            
                hyperp = utils.initialize_hyperparameters(model_key, config, default_hp, hp_ranges)
                feature_sets = utils.generate_feature_sets(features, config, X_train_.shape)
                # If shuffling is requested, perform label shuffling before training.
                for rss, random_seed_shuffle in enumerate(config['random_seeds_shuffle']):
                    if config['shuffle_labels']:
                        np.random.seed(int(random_seed_shuffle))
                        #For binary classification, swap half of the labels.
                        if problem_type == 'clf' and len(np.unique(y_train_)) == 2:
                            zero_indices = np.where(y_train_ == 0)[0]
                            one_indices = np.where(y_train_ == 1)[0]
                            zero_to_flip = np.random.choice(zero_indices, size=len(zero_indices) // 2, replace=False)
                            one_to_flip = np.random.choice(one_indices, size=len(one_indices) // 2, replace=False)
                            y_train_.iloc[zero_to_flip] = 1
                            y_train_.iloc[one_to_flip] = 0
                        else:
                            y_train_ = pd.Series(np.random.permutation(y_train_.values))

                        all_models = pd.read_csv(Path(str(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '')).replace('shuffle',''), f'all_models_{model_key}.csv'))

                        if config['shuffle_all']:
                            feature_names = [col for col in all_models.columns if any(x in col for x in dimension.split('__'))]
                            param_names = list(set(all_models.columns) - set(feature_names) - set(['threshold']))
                            hyperp = all_models[param_names]
                            feature_sets = []
                            for r,row in all_models.iterrows():
                                feature_sets.append([col for col in all_models.columns if col in feature_names and row[col] == 1])
                            
                            #Drop repeated feature sets
                            feature_sets = [list(x) for x in set(tuple(x) for x in feature_sets)]
                            hyperp = hyperp.drop_duplicates()     
                        else:
                            best_models_file_name = f"best_models_{scoring_metrics}_{config['kfold_folder']}_{config['stat_folder']}_{config['bootstrap_method']}_hyp_opt_feature_selection.csv".replace('__','_')
                            
                            if config['n_iter'] == 0:
                                best_models_file_name = best_models_file_name.replace('hyp_opt','no_hyp_opt')
                            if config['n_iter_features'] == 0:
                                best_models_file_name = best_models_file_name.replace('_feature_selection','')
                            
                            best_models = pd.read_csv(Path(results_dir,best_models_file_name))
                            best_models = best_models[best_models['y_label'] == y_label]
                            best_models = best_models[best_models['task'] == task]
                            best_models = best_models[best_models['dimension'] == dimension]

                            model_type = best_models['model_type'].values[0]
                            model_index = best_models['model_index'].values[0]

                            if model_type != model_key:
                                continue

                            feature_names = [col for col in all_models.columns if any(x in col for x in task.split('__'))]
                            param_names = list(set([col for col in all_models.columns if col not in feature_names]) - set(['threshold']))
                            hyperp = pd.DataFrame(all_models.loc[model_index][param_names]).T
                            feature_sets = [[col for col in all_models.columns if col in feature_names and all_models.loc[model_index][col] == 1]]
                    elif config['calibrate'] and Path(str(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '')).replace('shuffle',''), f'all_models_{model_key}.csv').exists():
                        all_models = pd.read_csv(Path(str(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '')).replace('shuffle',''), f'all_models_{model_key}.csv'))

                        feature_names = [col for col in all_models.columns if any(x in col for x in dimension.split('__'))]
                        param_names = list(set(all_models.columns) - set(feature_names) - set(['threshold']))
                        hyperp = all_models[param_names]
                        feature_sets = []
                        for r,row in all_models.iterrows():
                            feature_sets.append([col for col in all_models.columns if col in feature_names and row[col] == 1])
                        
                        feature_sets = [list(x) for x in set(tuple(x) for x in feature_sets)]
                        hyperp = hyperp.drop_duplicates() 
                        if 'priors' in hyperp.columns and hyperp['priors'].isna().all():
                            hyperp = hyperp.drop(columns='priors')
                        
                    # Check for data leakage.
                    assert set(ID_train_).isdisjoint(set(ID_test_)), 'Data leakage detected between train and test sets!'
                    
                    with open(Path(__file__).parent/'config.json', 'w') as f:
                        json.dump(config, f, indent=4)
                    if (Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', f'all_models_{model_key}.csv').exists() and config['calibrate'] == False) or (Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', f'cal_outputs_{model_key}.pkl').exists() and config['calibrate']):
                        if config['overwrite'] == False:
                            print(f'Results already exist for {task} - {y_label} - {model_key}. Skipping...')
                            continue
                    
                    print(f'Training model: {model_key}')

                    # Call CVT from utils to perform cross-validation training and tuning.
                    all_models, outputs_, cal_outputs_, X_dev_, y_dev_, IDs_dev_ = utils.CVT(
                        model=model_class,
                        scaler=(StandardScaler if config['scaler_name'] == 'StandardScaler' else MinMaxScaler),
                        imputer=KNNImputer,
                        X=X_train_,
                        y=y_train_,
                        iterator=CV,
                        random_seeds_train=config['random_seeds_train'],
                        hyperp=hyperp,
                        feature_sets=feature_sets,
                        IDs=ID_train_,
                        thresholds=thresholds,
                        parallel=config['parallel'],
                        problem_type=problem_type,
                        calmethod=calmethod,
                        calparams=calparams
                    )
                    
                    if rss == 0:
                        X_dev = np.expand_dims(X_dev_,axis=0)
                        y_dev = np.expand_dims(y_dev_,axis=0)
                        IDs_dev = np.expand_dims(IDs_dev_,axis=0)
                        outputs = np.expand_dims(outputs_,axis=0)
                        cal_outputs = np.expand_dims(cal_outputs_,axis=0)
                        X_test = np.expand_dims(X_test_,axis=0)
                        y_test = np.expand_dims(y_test_,axis=0)
                        IDs_test = np.expand_dims(ID_test_,axis=0)
                    else:
                        X_dev = np.concatenate((X_dev,np.expand_dims(X_dev_,axis=0)))
                        y_dev = np.concatenate((y_dev,np.expand_dims(y_dev_,axis=0)))
                        IDs_dev = np.concatenate((IDs_dev,np.expand_dims(IDs_dev_,axis=0)))
                        outputs = np.concatenate((outputs,np.expand_dims(outputs_,axis=0)))
                        cal_outputs = np.concatenate((cal_outputs,np.expand_dims(cal_outputs_,axis=0)))
                        X_test = np.concatenate((X_test,np.expand_dims(X_test_,axis=0)))
                        y_test = np.concatenate((y_test,np.expand_dims(y_test_,axis=0)))
                        IDs_test = np.concatenate((IDs_test,np.expand_dims(ID_test_,axis=0)))

                if outputs.shape[0] == 0:
                    continue

                # Save results.
                all_models.to_csv(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', f'all_models_{model_key}.csv'),index=False)

                result_files = {
                    'X_dev.pkl': X_dev,
                    'y_dev.pkl': y_dev,
                    'IDs_dev.pkl': IDs_dev,
                    f'outputs_{model_key}.pkl': outputs}
                
                if config['calibrate']:
                    result_files.update({f'cal_outputs_{model_key}.pkl': cal_outputs})

                if test_size > 0:
                    result_files.update({
                        'X_test.pkl': X_test_,
                        'y_test.pkl': y_test_,
                        'IDs_test.pkl': ID_test_,
                    })
                for fname, obj in result_files.items():
                    with open(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', fname), 'wb') as f:
                        pickle.dump(obj, f)
                
                with open(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', 'config.json'), 'w') as f:
                    json.dump(config, f, indent=4)
                logging.info(f'Results saved to {path_to_save}')
##----------------------------------------------------------------------------##
