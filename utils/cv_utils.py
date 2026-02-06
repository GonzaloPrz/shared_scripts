import pandas as pd
import numpy as np

from utils import Model

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from expected_cost.ec import *
from expected_cost.utils import *

import math, itertools

from joblib import Parallel, delayed

def nestedCVT(model_class,scaler,imputer,X,y,n_iter,iterator_outer,iterator_inner,strat_col,random_seeds_outer,hyperp_space,IDs,init_points=5,scoring='roc_auc',problem_type='clf',cmatrix=None,priors=None,threshold=None,feature_selection=True,parallel=True,calparams=None,calmethod=None,round_values=False,covariates=None,fill_na=None,regress_out_method='linear'):
    """
    Conducts nested cross-validation with recursive feature elimination (RFE) and hyperparameter tuning 
    to select and evaluate the best model configuration. Supports classification and regression tasks.

    Parameters
    ----------
    model_class : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    scaler : callable
        Scaler function to initialize and fit a scaler for feature preprocessing.
    imputer : callable
        Imputer function to initialize and fit an imputer for handling missing values.
    X : pd.DataFrame
        Input features data.
    y : pd.Series or np.array
        Target values.
    n_iter : int
        Number of iterations for hyperparameter tuning.
    iterator_outer : cross-validation generator
        Outer cross-validation iterator to split the dataset.
    iterator_inner : cross-validation generator
        Inner cross-validation iterator for tuning and feature selection.
    random_seeds_outer : list
        List of random seeds for reproducibility across outer folds.
    hyperp_space : dict
        Dictionary defining hyperparameter search space for model tuning.
    IDs : np.array
        Array of sample identifiers for tracking predictions across folds.
    scoring : str, optional
        Metric name for model selection (e.g., 'roc_auc_score') (default is 'roc_auc_score').
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').
    cmatrix : CostMatrix, optional
        Cost matrix for classification; defaults to None.
    priors : dict, optional
        Class priors for probability calibration in classification.
    threshold : float, optional
        Decision threshold for classification tasks.

    Returns
    -------
    all_models : pd.DataFrame
        DataFrame of all model configurations, including hyperparameters, feature selections, and scores.
    outputs_best : np.array
        Array of model outputs for each sample across configurations and random seeds.
    y_true : np.array
        Array of true target values for each sample across configurations.
    y_pred_best : np.array
        Array of predicted values across configurations and random seeds.
    IDs_val : np.array
        Array of IDs for samples used in predictions across outer folds.
        
    """
    
    if (cmatrix is None) & (problem_type == 'clf'):
        cmatrix = CostMatrix.zero_one_costs(K=len(np.unique(y)))

    features = X.columns.tolist()  # once, near the top
    
    iterator_inner.random_state = 42

    model_rfecv = model_class()

    if hasattr(model_rfecv,'kernel'):
        model_rfecv.kernel = 'linear'
    if hasattr(model_rfecv,'probability'):
        model_rfecv.probability = True
    if hasattr(model_rfecv,'random_state') and problem_type == 'clf':
        model_rfecv.random_state = int(42)

    def parallel_train(r,random_seed):
        models_r = pd.DataFrame(columns=['random_seed','fold','threshold',scoring] + list(hyperp_space.keys()) + list(features))
        iterator_outer.random_state = random_seed

        n_samples = X.shape[0] 

        outputs_best_r = np.full((n_samples,len(np.unique(y))),np.nan) if problem_type == 'clf' else np.full((n_samples),np.nan)

        y_true_r = np.full(n_samples,np.nan)

        y_pred_best_r = np.full(n_samples,np.nan)

        IDs_val_r = np.full(n_samples,fill_value='nan',dtype=object)
        
        model = Model(model_class,scaler,imputer,calmethod,calparams)
        for k,(train_index_out,test_index_out) in enumerate(iterator_outer.split(X,strat_col,IDs)): 
            X_dev, X_test = X.loc[train_index_out].reset_index(drop=True), X.loc[test_index_out].reset_index(drop=True)
            y_dev, y_test = y[train_index_out], y[test_index_out]
            if covariates is not None:
                covariates_dev= covariates.loc[train_index_out].reset_index(drop=True)
                covariates_test = covariates.loc[test_index_out].reset_index(drop=True)

            else:
                covariates_dev = None
                covariates_test = None

            y_true_r[test_index_out] = y_test
            
            IDs_dev = IDs.loc[train_index_out].reset_index(drop=True)
            
            IDs_val_r[test_index_out] = IDs.loc[test_index_out].reset_index(drop=True)

            #scaler_ = scaler().fit(X_dev)
            #imputer_ = imputer().fit(X_dev)

            #X_dev = pd.DataFrame(columns=X.columns,data=imputer_.transform(pd.DataFrame(columns=X_dev.columns,data=scaler_.transform(X_dev))))
            #X_dev  = X_dev.reindex(columns=features)
            #X_test = X_test.reindex(columns=features)
            #X_test = pd.DataFrame(columns=X.columns,data=imputer_.transform(pd.DataFrame(columns=X_test.columns,data=scaler_.transform(X_test))))
            print(f'Random seed {r+1}, Fold {k+1}')
            
            if n_iter > 0:
                best_params, best_score = tuning(model_class,scaler,imputer,X_dev,y_dev,IDs_dev,hyperp_space,iterator_inner,init_points=init_points,n_iter=n_iter,scoring=scoring,problem_type=problem_type,cmatrix=cmatrix,priors=priors,threshold=threshold,calmethod=calmethod,calparams=calparams,round_values=round_values,covariates=covariates_dev,fill_na=fill_na,regress_out_method=regress_out_method)
            else:
                best_params = model_class().get_params() 

            if 'n_estimators' in best_params.keys() and not isinstance(best_params['n_estimators'],type(None)):
                best_params['n_estimators'] = int(best_params['n_estimators'])
            elif 'n_neighbors' in best_params.keys() and not isinstance(best_params['n_neighbors'],type(None)):
                best_params['n_neighbors'] = int(best_params['n_neighbors'])
            elif 'max_depth' in best_params.keys() and not isinstance(best_params['max_depth'],type(None)):
                best_params['max_depth'] = int(best_params['max_depth'])
            if 'gpu_id' in best_params.keys() and not isinstance(best_params['gpu_id'],type(None)):
                best_params['gpu_id'] = None
            
            if hasattr(model_class(),'random_state') and model_class != SVR:
                best_params['random_state'] = int(42)
            if hasattr(model_class(),'probability') and model_class != SVR:
                best_params['probability'] = True

            if feature_selection:
                best_features, best_score = rfe(Model(model_class(**best_params),scaler,imputer,calmethod,calparams),X_dev,y_dev.values if isinstance(y_dev,pd.Series) else y_dev,IDs_dev,iterator_inner,scoring,problem_type,cmatrix=cmatrix,priors=priors,threshold=threshold,round_values=round_values,covariates=covariates_dev,fill_na=fill_na)
            else:
                best_features, best_score = X.columns, np.nan

            append_dict = {'random_seed':random_seed,'fold':k,'threshold':threshold,scoring:best_score}
            append_dict.update(best_params)
            append_dict.update({feature:1 if feature in best_features else 0 for feature in X_dev.columns}) 

            models_r.loc[len(models_r.index),:] = append_dict

            model = Model(model_class(**best_params),scaler,imputer,calmethod,calparams)
            model.train(X_dev[best_features],y_dev,covariates_dev,fill_na,regress_out_method)
            
            if problem_type == 'clf':
                outputs_best_ = model.eval(X_test[best_features],problem_type,covariates_test,fill_na)
                if isinstance(threshold,float) & (len(np.unique(y)) == 2):
                    y_pred_best_ = [1 if x > threshold else 0 for x in outputs_best_[:,1]]
                else:
                    y_pred_best_= bayes_decisions(scores=outputs_best_,costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
                
                y_pred_best_r[test_index_out] = np.round(y_pred_best_,decimals=0) if round_values else y_pred_best_

            else:
                outputs_best_ = model.eval(X_test[best_features],problem_type,covariates_test,fill_na)
                y_pred_best_r[test_index_out] = np.round(outputs_best_,decimals=0) if round_values else outputs_best_
            outputs_best_r[test_index_out] = outputs_best_

        outputs_best_r = outputs_best_r[~np.isnan(outputs_best_r).all(axis=1)] if problem_type == 'clf' else outputs_best_r[~np.isnan(outputs_best_r)]
        y_pred_best_r = y_pred_best_r[~np.isnan(y_pred_best_r)]
        y_true_r = y_true_r[~np.isnan(y_true_r)]
        IDs_val_r = IDs_val_r[IDs_val_r != 'nan']

        return models_r,outputs_best_r,y_true_r,y_pred_best_r,IDs_val_r
    
    results = Parallel(n_jobs=-1 if parallel else 1)(delayed(parallel_train)(r,random_seed_train) for (r,random_seed_train) in enumerate(random_seeds_outer))
    all_models = pd.concat([result[0] for result in results],ignore_index=True,axis=0)
    outputs_best = np.concatenate(([np.expand_dims(result[1],axis=0) for result in results]),axis=0)
    y_true = np.concatenate(([np.expand_dims(result[2],axis=0) for result in results]),axis=0)
    y_pred_best = np.concatenate(([np.expand_dims(result[3],axis=0) for result in results]),axis=0)
    IDs_val = np.concatenate(([np.expand_dims(result[4],axis=0) for result in results]),axis=0)

    return all_models,outputs_best,y_true,y_pred_best,IDs_val

def initialize_hyperparameters(model_key,config,default_hp,hp_range):
    """
    Initialize hyperparameter DataFrame for a given model.
    In a production system this might be replaced by loading a pre-tuned configuration.
    """
    # Default hyperparameters (as a pandas DataFrame)

    n = 0
    hp = pd.DataFrame(default_hp.get(model_key),index=[0])

    while hp.shape[0] < config['n_iter']+1 and n < 1000:        
        # If no default hyperparameters are available, generate random hyperparameters
        np.random.seed(n)
        new_hp = {key: np.random.choice(hp_range[model_key][key]) for key in hp_range[model_key].keys()}
        
        hp = pd.concat([hp, pd.DataFrame(new_hp,index=[0])], ignore_index=True)

        #Drop duplicates:
        hp = hp.drop_duplicates()
        n += 1

    return hp

def generate_feature_sets(features, config, data_shape):
    """
    Generate a list of feature subsets for evaluation.
    Either compute all combinations up to a maximum length or generate a random sample.
    """
    if (config["n_folds"] < 1) & (config["n_folds"] > 0):
        n_possible = int(config["feature_sample_ratio"] * data_shape[0] * (1 - config["test_size"]) * (1 - config["n_folds"])) - 1
    else:
        n_possible = int(config["feature_sample_ratio"] * data_shape[0] * (1 - config["test_size"]) * ((config["n_folds"] - 1) / config["n_folds"])) - 1
    # Determine total number of combinations.
    num_comb = sum(math.comb(len(features), k+1) for k in range(len(features)-1))
    feature_sets = []
    if config["n_iter_features"] > num_comb:
        for k in range(len(features)-1):
            for comb in itertools.combinations(features, k+1):
                feature_sets.append(list(comb))
    else:
        for _ in range(int(config["n_iter_features"])):
            # Use np.random.choice without replacement
            n_iter = 0
            np.random.seed(n_iter)
            new_set = list(np.random.choice(features, np.min((len(features),n_possible)), replace=True))
            #Eliminate duplicates
            new_set = list(set(new_set))
            while sorted(new_set) in feature_sets and n_iter < 100:
                n_iter += 1
                np.random.seed(n_iter)
                new_set = list(set(np.random.choice(features, np.min((len(features),n_possible)), replace=True)))
            feature_sets.append(sorted(new_set))    
            
    # Always include the full feature set.
    feature_sets.append(list(features))
    
    # Ensure that the feature sets are unique.
    feature_sets = list(set([tuple(set(feature_set)) for feature_set in feature_sets]))
    feature_sets = [list(feature_set) for feature_set in feature_sets]
    return feature_sets

def CV(model_class, params, scaler, imputer, X, y, feature_set,all_features, threshold, iterator, random_seeds_train, IDs, problem_type='clf',calmethod=None,calparams=None,fill_na=None):
    """
    Cross-validation function to train and evaluate a model with specified parameters, 
    feature engineering, and evaluation metrics. Supports both classification and regression.

    Parameters
    ----------
    model_class : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    params : dict
        Parameters to initialize the model.
    scaler : object
        Scaler instance to preprocess the feature data.
    imputer : object
        Imputer instance to handle missing values.
    X : pd.DataFrame
        Input features data.
    y : pd.Series or np.array
        Target values.
    all_features : list
        List of all possible feature names, marking presence or absence for feature engineering.
    threshold : float
        Decision threshold for classification tasks.
    iterator : cross-validation generator
        Cross-validation iterator to split the dataset.
    random_seeds_train : list
        List of random seeds for reproducibility in training and evaluation.
    IDs : np.array
        Array of sample identifiers for tracking predictions.
    cmatrix : CostMatrix, optional
        Cost matrix for classification; defaults to [[0,1],[1,0]] if not provided.
    priors : dict, optional
        Class priors for probability calibration in classification.
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').

    Returns
    -------
    model_params : pd.DataFrame
        DataFrame of model parameters used for training.
    outputs_dev : np.array
        Array of model outputs per cross-validation fold for each sample.
    y_dev : np.array
        Array of true target values across folds.
    y_pred : np.array
        Array of predicted values across folds.
    IDs_dev : np.array
        Array of IDs for samples used in predictions across folds.
    """

    model_params = params.copy()
    features = {feature: 0 for feature in all_features}
    features.update({feature: 1 for feature in feature_set})
    model_params.update(features)

    model_params = pd.DataFrame(model_params, index=[0])

    n_seeds = len(random_seeds_train)
    if hasattr(iterator,'test_size'):
        n_samples = int(X.shape[0]*iterator.test_size)
    else:
        n_samples = X.shape[0]

    n_features = X.shape[1]
    
    n_classes = len(np.unique(y))

    X_dev = np.empty((n_seeds, n_samples, n_features))
    y_dev = np.empty((n_seeds, n_samples))
    IDs_dev = np.empty((n_seeds, n_samples), dtype=object)
    outputs_dev = np.empty((n_seeds, n_samples, n_classes)) if problem_type == 'clf' else np.empty((n_seeds, n_samples))
    cal_outputs_dev = np.empty_like(outputs_dev)

    iterator.random_state = 42

    for r, random_seed in enumerate(random_seeds_train):
        iterator.random_state = random_seed

        for train_index, test_index in iterator.split(X, y):
            model = Model(model_class(**params), scaler, imputer,calmethod,calparams)
            if hasattr(model.model, 'random_state'):
                model.model.random_state = 42
                
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index].values.squeeze()
            y_test = y.iloc[test_index].values.squeeze()
            IDs_test = IDs[test_index]
            try:
                X_dev[r, test_index] = X_test
                y_dev[r, test_index] = y_test
                IDs_dev[r, test_index] = IDs_test
                X_train = X_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                IDs_dev[r, test_index] = IDs_test
            except:
                X_dev[r, :] = X.iloc[test_index]
                y_dev[r, :] = y[test_index].squeeze()
                IDs_dev[r, :] = IDs[test_index]
                
            model.train(X_train[feature_set],y_train,fill_na)
            try:
                outputs_dev[r, test_index] = model.eval(X_test[feature_set], problem_type)
                if calmethod is not None:
                    cal_outputs_dev[r, test_index],_ = model.calibrate(outputs_dev[r,test_index],y_dev[r,test_index])
                else:
                    cal_outputs_dev[r, test_index] = outputs_dev[r, test_index]
            except:
                outputs_dev[r, :] = model.eval(X_test[feature_set], problem_type)

    if problem_type == 'clf':
        model_params['threshold'] = threshold

    return model_params, outputs_dev, cal_outputs_dev, X_dev, y_dev, IDs_dev

def CVT(model, scaler, imputer, X, y, iterator, random_seeds_train, hyperp, feature_sets, IDs, thresholds=[None], parallel=True, problem_type='clf',calmethod=None,calparams=None):
    """
    Cross-validation testing function for model training and evaluation with hyperparameter 
    tuning, feature set selection, and parallel processing options. Supports classification 
    and regression tasks.

    Parameters
    ----------
    model : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    scaler : object
        Scaler instance to preprocess feature data.
    imputer : object
        Imputer instance to handle missing values.
    X : pd.DataFrame
        Input features data.
    y : pd.Series or np.array
        Target values.
    iterator : cross-validation generator
        Cross-validation iterator to split the dataset.
    random_seeds_train : list
        List of random seeds for reproducibility in training and evaluation.
    hyperp : pd.DataFrame
        DataFrame of hyperparameter values for each model configuration.
    feature_sets : list of lists
        List of feature subsets to evaluate, each list containing feature names.
    IDs : np.array
        Array of sample identifiers for tracking predictions.
    thresholds : list, optional
        List of decision thresholds for classification; defaults to [None].
    cmatrix : CostMatrix, optional
        Cost matrix for classification; defaults to [[0,1],[1,0]] if not provided.
    priors : dict, optional
        Class priors for probability calibration in classification.
    parallel : bool, optional
        If True, enables parallel processing for cross-validation tasks (default is True).
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').

    Returns
    -------
    all_models : pd.DataFrame
        DataFrame of all model configurations, including hyperparameters and feature sets.
    all_outputs : np.array
        Array of model outputs for all pooled samples across each configuration and random 
        seed in cross-validation.
    all_y_pred : np.array
        Array of predicted values across configurations and folds.
    y_true : np.array
        Array of true target values.
    IDs_dev : np.array
        Array of IDs for samples used in predictions across folds.
    """
    
    features = X.columns
    if not isinstance(thresholds,list):
        thresholds = [thresholds]

    if hasattr(model(), 'random_state') and model != SVR:
        hyperp['random_state'] = 42
    
    all_models = pd.DataFrame(columns=list(hyperp.columns) + list(features),index=range(hyperp.shape[0]*len(feature_sets)*len(thresholds)))
    if hasattr(iterator,'test_size'):
        n_samples = int(X.shape[0]*iterator.test_size)
    else:
        n_samples = X.shape[0]
    all_outputs = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), n_samples, len(np.unique(y)))) if problem_type == 'clf' else np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), n_samples))

    all_cal_outputs = np.empty_like(all_outputs)
    X_dev = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), n_samples, X.shape[1]))
    y_true = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), n_samples))
    IDs_dev = np.empty((hyperp.shape[0]*len(feature_sets)*len(thresholds), len(random_seeds_train), n_samples), dtype=object)

    def process_combination(c,f,t,problem_type,calmethod,calparams):
        params = hyperp.iloc[c, :].to_dict()
        return c, f, t, CV(model, params, scaler, imputer, X, y, feature_sets[f],features, thresholds[t], iterator, [int(seed) for seed in random_seeds_train], IDs, problem_type,calmethod,calparams)
        
    if parallel:
        results = Parallel(n_jobs=-1,timeout=300)(delayed(process_combination)(c, f,threshold,problem_type,calmethod,calparams) for c, f, threshold in itertools.product(range(hyperp.shape[0]), range(len(feature_sets)), range(len(thresholds))))
        for c,f,t, result in results:
            all_models.loc[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[0].iloc[0]
            all_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t,] = result[1]
            all_cal_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[2]
            X_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[3]
            y_true[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[4]
            IDs_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[5]
    else:
        for c,f,t in itertools.product(range(hyperp.shape[0]), range(len(feature_sets)), range(len(thresholds))):
            _,_,_, result = process_combination(c, f, t,problem_type,calmethod,calparams)
            all_models.loc[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[0].iloc[0]
            all_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[1]
            all_cal_outputs[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[2]
            X_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[3]
            y_true[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[4]
            IDs_dev[c*len(feature_sets)*len(thresholds)+f*len(thresholds)+t] = result[5]
    
    return all_models, all_outputs, all_cal_outputs, X_dev[0], y_true[0], IDs_dev[0]

