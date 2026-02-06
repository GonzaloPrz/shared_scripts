import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit, GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier as xgboost
from xgboost import XGBRegressor as xgboostr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import itertools, json
import logging,sys,os,argparse
from psrcal.calibration import AffineCalLogLoss
from sklearn.preprocessing import LabelEncoder

from src.utils.utils import *
from src.utils.cv_utils import *
from src.utils.utils import PROJECT_ROOT

##---------------------------------PARAMETERS---------------------------------##
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train models with hyperparameter optimization and feature selection'
    )
    parser.add_argument('--project_name', default='affective_pitch_CN_FTD',type=str,help='Project name')
    parser.add_argument('--stats', type=str, default='', help='Stats to be considered (default = all)')
    parser.add_argument('--shuffle_labels', type=int, default=0, help='Shuffle labels flag (1 or 0)')
    parser.add_argument('--stratify', type=int, default=1, help='Stratification flag (1 or 0)')
    parser.add_argument('--calibrate', type=int, default=0, help='Whether to calibrate models')
    parser.add_argument('--n_folds_outer', type=float, default=5, help='Number of folds for cross validation (outer loop)')
    parser.add_argument('--n_folds_inner', type=float, default=0.2, help='Number of folds for cross validation (inner loop)')
    parser.add_argument('--init_points', type=int, default=20, help='Number of random initial points to test during Bayesian optimization')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of hyperparameter iterations')
    parser.add_argument('--feature_selection',type=int,default=1,help='Whether to perform feature selection with RFE or not')
    parser.add_argument('--fill_na',type=int,default=-10,help='Values to fill nan with. Default (=0) means no filling (imputing instead)')
    parser.add_argument('--n_seeds_train',type=int,default=5,help='Number of seeds for cross-validation training')
    parser.add_argument('--scaler_name', type=str, default='StandardScaler', help='Scaler name')
    parser.add_argument('--id_col', type=str, default='id', help='ID column name')
    parser.add_argument('--n_boot',type=int,default=1000,help='Number of bootstrap iterations')
    parser.add_argument('--n_boot_train',type=int,default=0,help='Number of bootstrap iterations for training')
    parser.add_argument('--n_boot_test',type=int,default=1000,help='Number of bootstrap iterations for testing')
    parser.add_argument('--filter_outliers',type=int,default=0,help='Whether to filter outliers in regression problems')
    parser.add_argument('--early_fusion',type=int,default=0,help='Whether to perform early fusion')
    parser.add_argument('--overwrite',type=int,default=0,help='Whether to overwrite past results or not')
    parser.add_argument('--parallel',type=int,default=0,help='Whether to parallelize processes or not')
    parser.add_argument('--n_seeds_test',type=int,default=1,help='Number of seeds for testing')
    parser.add_argument('--bootstrap_method',type=str,default='bca',help='Bootstrap method [bca, percentile, basic]')
    parser.add_argument('--round_values',type=int,default=0,help='Whether to round predicted values for regression or not')
    parser.add_argument('--add_dem',type=int,default=0,help='Whether to add demographic features or not')
    parser.add_argument('--cut_values',type=float,default=-1,help='Cut values above a given threshold')
    parser.add_argument('--regress_out',type=str,default='',help='List of demographic variables to regress out from target variable, separated by "_"')
    parser.add_argument('--regress_out_method',type=str,default='linear',help='Whether to perform linear or non-linear regress-out'),
    parser.add_argument('--scoring', type=str, default='', help='Scoring method for model selection')
    return parser.parse_args()

def load_configuration(args):
    # Global configuration dictionaries
    config = dict(
        project_name = args.project_name,
        stats = str(args.stats),
        shuffle_labels = bool(args.shuffle_labels),
        stratify = bool(args.stratify),
        calibrate = bool(args.calibrate),
        n_folds_outer = float(args.n_folds_outer),
        n_folds_inner = float(args.n_folds_inner),
        n_iter = float(args.n_iter),
        feature_selection = bool(args.feature_selection),
        fill_na = int(args.fill_na),
        init_points = float(args.init_points),
        n_seeds_train = float(args.n_seeds_train) if args.n_folds_outer!= -1 else float(1),
        scaler_name = args.scaler_name,
        id_col = args.id_col,
        n_boot = float(args.n_boot),
        n_boot_test = float(args.n_boot_test),
        n_boot_train = float(args.n_boot_train),
        filter_outliers = bool(args.filter_outliers),
        early_fusion = bool(args.early_fusion),
        overwrite = bool(args.overwrite),
        parallel = bool(args.parallel),
        n_seeds_test = float(args.n_seeds_test) if args.n_folds_outer!= -1 else float(0),
        bootstrap_method = args.bootstrap_method,
        round_values = bool(args.round_values),
        add_dem = bool(args.add_dem),
        cut_values = float(args.cut_values),
        regress_out = list(set(sorted(list(args.regress_out.split('_')))) - set([''])),
        regress_out_method = str(args.regress_out_method),
        scoring = str(args.scoring)
    )

    return config

imputer = KNNImputer

args = parse_args()
config = load_configuration(args)
project_name = config['project_name']
add_dem = config['add_dem']
round_values = config['round_values']
cut_values = config['cut_values']
regress_out = config['regress_out']
fill_na = config['fill_na'] if config['fill_na'] != 0 else None

logging.info('Configuration loaded. Starting training...')
logging.info('Training completed.')

##------------------ Configuration and Parameter Parsing ------------------##
home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    data_dir = home / 'data' / project_name
else:
    data_dir = Path('D:/CNC_Audio/gonza/data', project_name)

results_dir = Path(str(data_dir).replace('data', 'results'))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

main_config = json.load(open(Path(PROJECT_ROOT,'config','main_config.json')))

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]

try:
    covars = main_config['covariates'][project_name]
except:
    covars = []

try:
    test_size = main_config['test_size'][project_name]
except:
    test_size = 0

try:
    thresholds = main_config['thresholds'][project_name]
except:
    thresholds = [None]

try:
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
except:
    cmatrix = None

config['test_size'] = float(test_size)
config['data_file'] = data_file
config['tasks'] = tasks
config['single_dimensions'] = single_dimensions    

config['y_labels'] = y_labels
config['avoid_stats'] = list(set(['min','max','median','skew','kurt','std','mean']) - set(config['stats'].split('_'))) if config['stats'] != '' else []
config['stat_folder'] = '_'.join(sorted(config['stats'].split('_')))

config['random_seeds_train'] = [int(3**x) for x in np.arange(1, config['n_seeds_train']+1)]
config['random_seeds_test'] = [int(3**x) for x in np.arange(1, config['n_seeds_test']+1)] if config['test_size'] > 0 else ['']
config['bayes'] = True

if config['calibrate']:
    calmethod = AffineCalLogLoss
    calparams = {'bias':True, 'priors':None}
else:
    calmethod = None
    calparams = None

config['covariates'] = covars

models_dict = {'clf':{
                    'lr':LR,
                    'knnc':KNNC,
                    'xgb':xgboost,
                    #'rf':RandomForestClassifier
                    #'qda':QDA,
                    #'lda': LDA
                    },
                
                'reg':{'lasso':Lasso,
                    'ridge':Ridge,
                    'elastic':ElasticNet,
                    #'rf':RandomForestRegressor,
                    'knnr':KNNR,
                    'svr':SVR,
                    #'xgb':xgboostr
                    }
}

hyperp = json.load(Path(PROJECT_ROOT,'config','hyperparameters.json').open())

for task in tasks:
    if isinstance(y_labels,dict):
        y_labels_ = y_labels[task]
    else:
        y_labels_ = y_labels
    
    for y_label in y_labels_:
        dimensions = list()
        if isinstance(single_dimensions,dict):
            single_dimensions_ = single_dimensions[task]
        else:
            single_dimensions_ = single_dimensions

        if isinstance(single_dimensions_,list) and config["early_fusion"]:
            for ndim in range(len(single_dimensions_)):
                for dimension in itertools.combinations(single_dimensions_,ndim+1):
                    dimensions.append('__'.join(dimension))
        elif isinstance(single_dimensions_,list) and not config["early_fusion"]:
            dimensions = single_dimensions_
        else:
            dimensions = [single_dimensions_]

        for dimension in dimensions:            
            try:
                all_data = pd.read_csv(Path(data_dir,data_file))
            except:
                all_data = pd.read_csv(Path(data_dir,data_file),encoding='latin1')

            all_data = all_data.loc[:, ~all_data.columns.str.match(r'^Unnamed')]

            features = [col for col in all_data.columns if any(f'{x}__{y}' in col for x,y in itertools.product(task.split('__'),dimension.split('__'))) and 'timestamp' not in col]

            if len(config["avoid_stats"]) > 0:
                features = [col for col in features if all(f'_{x}' not in col for x in config['avoid_stats'])]
            
            covars_regress_out = list(set(regress_out).intersection(set(all_data.columns)))
            config['regress_out'] = covars_regress_out

            if config['add_dem']:
                for col in set(['sex','age','education','handedness']).intersection(set(all_data.columns)):
                    
                    all_data[f'{task}__dem__{col}'] = all_data[col]

                demographic_features = [f'{task}__dem__{col}' for col in all_data.columns if col in ['sex','age','education','handedness']]

                features.extend(demographic_features)
                dimension = dimension + '__dem' if dimension != '' else 'dem'
            
            print(task,dimension)
            if len(covars + covars_regress_out) != 0:
                all_data = all_data.dropna(subset=covars + covars_regress_out,how='any').reset_index(drop=True)
            
            if len(np.unique(all_data[y_label])) > 3:
                config['problem_type'] = 'reg'
                if config['scoring'] != '':
                    scoring_metric = config['scoring']
                else:
                    scoring_metric = 'r2'
            else:
                config['problem_type'] = 'clf'
                if config['scoring'] != '':
                    scoring_metric = config['scoring']
                else:
                    scoring_metric = 'roc_auc' if len(np.unique(all_data[y_label])) == 2 else 'norm_expected_cost'

            if config['problem_type'] == 'reg' and config['filter_outliers']:
                all_data = filter_outliers(all_data,parametric=True,n_sd=2)

            covariates_ = all_data[[config['id_col']] + covars]
            covariates_regress_out = all_data[[config['id_col']] + covars_regress_out]
                    
            data = all_data[features + [y_label, config['id_col']]]
            #data.dropna(subset=[col for col in data.columns if data[col].isna().sum()/data.shape[0] > 0.20], axis=1,inplace=True)
            data.dropna(subset=y_label,inplace=True)
            if cut_values > 0:
                data = data[data[y_label] <= cut_values]
            
            data = data.reset_index(drop=True)
    
            #convert y_label to categories
            y = data.pop(y_label)

            y = pd.Series(LabelEncoder().fit_transform(y) if config['problem_type'] == 'clf' else y,name=y_label)

            if data.shape[0] == 0:
                continue
            
            if config['shuffle_labels'] and config['problem_type'] == 'clf':
                np.random.seed(42)
                zero_indices = np.where(y == 0)[0]
                one_indices = np.where(y == 1)[0]

                # Shuffle and select half of the indices for flipping
                zero_to_flip = np.random.choice(zero_indices, size=len(zero_indices) // 2, replace=False)
                one_to_flip = np.random.choice(one_indices, size=len(one_indices) // 2, replace=False)

                # Flip the values at the selected indices
                y[zero_to_flip] = 1
                y[one_to_flip] = 0

            elif config['shuffle_labels']:
                np.random.seed(42)
                #Perform random permutations of the labels
                y = np.random.permutation(y)
                    
            ID = data.pop(config['id_col'])
            
            covariates_ = covariates_[covariates_[config['id_col']].isin(np.unique(ID))].reset_index(drop=True) if covariates_.shape[1] > 0 else None
            covariates_regress_out = covariates_regress_out[covariates_regress_out[config['id_col']].isin(np.unique(ID))].reset_index(drop=True) if covariates_regress_out.shape[1] > 0 else None

            covariates_regress_out.drop(config['id_col'],axis=1,inplace=True) if covariates_regress_out is not None else None
            covariates_.drop(config['id_col'],axis=1,inplace=True) if covariates_ is not None else None

            if (config['problem_type'] == 'reg') & ('group' in data.columns) & (config['stratify']):
                strat_col = data.pop('group')
            elif (config['problem_type'] == 'clf') & (config['stratify']):
                strat_col = y
            else:
                strat_col = None
            
            for covariate in covars + covars_regress_out:
                if not isinstance(all_data[covariate],(int,float)):
                    all_data[covariate] = LabelEncoder().fit_transform(all_data[covariate])
                    
            for model_key, model_class in models_dict[config['problem_type']].items():        
                print(model_key)
                
                held_out = float(config["test_size"]) > 0
                n_folds_outer = config['n_folds_outer']
                n_folds_inner = config['n_folds_inner']

                if held_out:
                    n_samples_dev = int(data.shape[0] * (1 - config['test_size']))
                    random_seeds_test = config['random_seeds_test']
                else:
                    n_samples_dev = data.shape[0]
                    random_seeds_test = ['']
                    config["n_seeds_test"] = 0
                    config["random_seeds_test"] = ['']

                if n_folds_outer== 0:
                    n_folds_outer= int(n_samples_dev / np.unique(y).shape[0])
                    CV_outer = (StratifiedGroupKFold(n_splits=n_folds_outer, shuffle=True)
                                if strat_col is not None 
                                else GroupKFold(n_splits=n_folds_outer, shuffle=True,))
                    config["kfold_folder"] = f'l{np.unique(y).shape[0]}out'
                    n_samples_outer = n_samples_dev - np.unique(y).shape[0]
                elif n_folds_outer== -1:
                    CV_outer = LeaveOneGroupOut()
                    n_samples_outer = n_samples_dev - 1
                    config["kfold_folder"] = 'loocv'
                elif n_folds_outer < 1:
                    CV_outer = (StratifiedShuffleSplit(n_splits=1,test_size=n_folds_outer)
                                if strat_col is not None
                                else GroupShuffleSplit(n_splits=1,test_size=n_folds_outer))
                    n_samples_outer = int(n_samples_dev*(1-n_folds_outer))
                    config['kfold_folder'] = f'{int(n_folds_outer*100)}pct'

                else:
                    n_folds_outer = int(n_folds_outer)
                    CV_outer = (StratifiedGroupKFold(n_splits=n_folds_outer, shuffle=True)
                                if strat_col is not None
                                else GroupKFold(n_splits=n_folds_outer, shuffle=True))
                    n_samples_outer = int(n_samples_dev*(1-1/n_folds_outer))
                    config['kfold_folder'] = f'{n_folds_outer}_folds' 
                
                if n_folds_inner == 0:
                    n_folds_inner = int(n_samples_outer / np.unique(y).shape[0])
                    CV_inner = (StratifiedGroupKFold(n_splits=n_folds_inner, shuffle=True)
                                if strat_col is not None
                                else GroupKFold(n_splits=n_folds_inner, shuffle=True))
                    config["kfold_folder"] += f'_l{np.unique(y).shape[0]}ocv'
                    n_max = n_samples_outer - np.unique(y).shape[0]
                elif n_folds_inner == -1:
                    CV_inner = LeaveOneGroupOut()
                    config["kfold_folder"] += '_loocv'
                    n_max = n_samples_outer - 1
                elif n_folds_inner < 1:
                    CV_inner = (StratifiedShuffleSplit(n_splits=1,test_size=n_folds_inner)
                                if strat_col is not None
                                else GroupShuffleSplit(n_splits=1,test_size=n_folds_inner))
                    n_max = int(n_samples_outer*(1-n_folds_inner))
                    config["kfold_folder"] += f'_{int(n_folds_inner*100)}pct'
                else: 
                    n_folds_inner = int(n_folds_inner)
                    CV_inner = (StratifiedGroupKFold(n_splits=n_folds_inner, shuffle=True)
                                if strat_col is not None
                                else GroupKFold(n_splits=n_folds_inner, shuffle=True))
                    n_max = int(n_samples_outer*(1-1/n_folds_inner))
                    config["kfold_folder"] += f'_{n_folds_inner}_folds'

                with open(PROJECT_ROOT/'config'/'config.json', 'w') as f:
                    json.dump(config, f, indent=4)

                for random_seed_test in random_seeds_test:
                    
                    subfolders = [
                    task, dimension,
                    config['kfold_folder'], f'{y_label}_res_{config["regress_out_method"]}' if len(regress_out) > 0 else y_label, config['stat_folder'],scoring_metric,
                    'hyp_opt' if config['n_iter'] > 0 else '','feature_selection' if config['feature_selection'] else '',
                    'filter_outliers' if config['filter_outliers'] and config['problem_type'] == 'reg' else '','rounded' if round_values else '','cut' if cut_values > 0 else '',
                    'shuffle' if config['shuffle_labels'] else '', f'random_seed_{int(random_seed_test)}' if config['test_size'] else ''
                    ]
                    
                    path_to_save = results_dir.joinpath(*[str(s) for s in subfolders if s])
                    path_to_save.mkdir(parents=True, exist_ok=True)

                    versions_dir = [folder.name for folder in path_to_save.iterdir() if folder.is_dir() and folder.name.startswith('v_')]

                    if len(versions_dir) == 0:
                        version = 1
                    else:
                        existing_versions = [folder.name for folder in path_to_save.iterdir() if folder.is_dir() and folder.name.startswith('v_')]

                        for existing_version in existing_versions:
                            try:
                                data_file_ = json.load(open(Path(path_to_save,existing_version,'config.json')))['data_file']
                            except:
                                data_file_ = data_file
                                config['data_file'] = data_file_
                                
                            if data_file_ == data_file:
                                version = int(existing_version.split('_')[1])
                                break       
                        else:
                            version = max([int(v.split('_')[1]) for v in existing_versions]) + 1

                    config['version'] = f'v_{version}'
                    path_to_save = Path(path_to_save,config['version'])
                    path_to_save.mkdir(exist_ok=True,parents=True)

                    if Path(path_to_save,'config.json').exists():
                        with open(Path(path_to_save,'config.json'), 'rb') as f:
                            old_config = json.load(f)
                            old_config['n_boot'] = config['n_boot']

                        if (not config['overwrite']) & (any(old_config[x] != config[x] for x in ['n_iter','init_points','n_seeds_train','n_boot'])):
                            for key in ['n_iter','init_points','n_seeds_train','n_boot']:
                                print(f'Warning: {key} has changed from {old_config[key]} to {config[key]}. Overwriting previous results.')
                                config[key] = old_config[key]
                        with open(Path(path_to_save,'config.json'),'w') as f:
                            json.dump(config, f, indent=4)
                            
                    if 'scoring_metric' not in list(config.keys()):
                        config['scoring_metric'] = scoring_metric
                        config['add_dem'] = add_dem
                        config['round_values'] = round_values

                    with open(Path(Path(__file__).parent,'config.json'),'w') as f:
                        json.dump(config, f, indent=4)

                    if test_size > 0:
                        X_train_, X_test_, y_train_, y_test_, ID_train_, ID_test_ = train_test_split(
                            data, y, ID,
                            test_size=config['test_size'],
                            random_state=int(random_seed_test),
                            shuffle=True,
                            stratify=strat_col)
                        
                        if strat_col is not None:
                            strat_col_train_, strat_col_test_ = train_test_split(
                            strat_col,
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
                        if strat_col is not None:
                            strat_col_train_ = strat_col_train_.reset_index(drop=True)
                        else:
                            strat_col_train_ = None

                    else:
                        X_train_, y_train_, ID_train_ = data.reset_index(drop=True), y.reset_index(drop=True), ID.reset_index(drop=True)
                        if strat_col is not None:
                            strat_col_train_ = strat_col.reset_index(drop=True)
                        else:
                            strat_col_train_ = None
                            
                        X_test_, y_test_, ID_test_ = pd.DataFrame(), pd.Series(), pd.Series()

                    data_train = pd.concat((X_train_,y_train_,ID_train_),axis=1)

                    data_test = pd.concat((X_test_,y_test_,ID_test_),axis=1)
                    
                    data_train.to_csv(Path(path_to_save,'data_train.csv'),index=False)
                    data_test.to_csv(Path(path_to_save,'data_test.csv'),index=False)

                    hyperp['knnc']['n_neighbors'] = (1,n_max)
                    hyperp['knnr']['n_neighbors'] = (1,n_max)

                    # Check for data leakage.
                    assert set(ID_train_).isdisjoint(set(ID_test_)), 'Data leakage detected between train and test sets!'

                    if (Path(path_to_save,f'all_models_{model_key}.csv').exists() and config['calibrate'] == False) or (Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', f'cal_outputs_{model_key}.pkl').exists() and config['calibrate']):
                        if not bool(config['overwrite']):
                            print(f'Results already exist for {task} - {y_label} - {model_key}. Skipping...')
                            continue
                    
                    print(f'Training model: {model_key}')

                    all_models,outputs_best,y_dev,y_pred_best,IDs_dev = utils.nestedCVT(model_class=models_dict[config['problem_type']][model_key],
                                                                                        scaler=StandardScaler if config['scaler_name'] == 'StandardScaler' else MinMaxScaler,
                                                                                        imputer=imputer,
                                                                                        X=X_train_,
                                                                                        y=y_train_.values if isinstance(y_train_, pd.Series) else y_train_,
                                                                                        n_iter=int(config['n_iter']),
                                                                                        iterator_outer=CV_outer,
                                                                                        iterator_inner=CV_inner,
                                                                                        strat_col=strat_col_train_,
                                                                                        random_seeds_outer=config['random_seeds_train'],
                                                                                        hyperp_space=hyperp[model_key],
                                                                                        IDs=ID_train_,
                                                                                        init_points=int(config['init_points']),
                                                                                        scoring=scoring_metric,
                                                                                        problem_type=config['problem_type'],
                                                                                        cmatrix=cmatrix,priors=None,
                                                                                        threshold=thresholds,
                                                                                        parallel=bool(config['parallel']),
                                                                                        feature_selection=bool(config['feature_selection']),
                                                                                        calmethod=calmethod,
                                                                                        calparams=calparams,
                                                                                        round_values=round_values,
                                                                                        covariates=covariates_regress_out if covariates_regress_out.shape[1] > 0 else None,
                                                                                        fill_na = fill_na,
                                                                                        regress_out_method = config['regress_out_method']
                                                                                        )
                
                    with open(Path(path_to_save,'config.json'),'w') as f:
                        json.dump(config,f)
                
                    all_models.to_csv(Path(path_to_save,f'all_models_{model_key}.csv'),index=False)
                    result_files = {
                        'X_train.npy': X_train_,
                        'y_train.npy': y_train_,
                        'IDs_train.npy': ID_train_,
                        'y_dev.npy': y_dev,
                        'IDs_dev.npy': IDs_dev,
                        f'outputs_{model_key}.npy': outputs_best}
                    
                    if test_size > 0:
                        result_files.update({
                            'X_test.npy': X_test_,
                            'y_test.npy': y_test_,
                            'IDs_test.npy': ID_test_,
                        })
                    for fname, obj in result_files.items():
                        with open(Path(path_to_save, fname), 'wb') as f:
                            np.save(f, obj)
                    
                    with open(Path(path_to_save, 'config.json'), 'w') as f:
                        json.dump(config, f, indent=4)