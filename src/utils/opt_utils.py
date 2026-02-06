import numpy as np
import pandas as pd

from expected_cost.ec import *
from expected_cost.utils import *

from psrcal.losses import LogLoss

from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score

from bayes_opt import BayesianOptimization

from src.utils.utils import Model

import torch

def rfe(model, X, y, groups, iterator, scoring='roc_auc', problem_type='clf',cmatrix=None,priors=None,threshold=None,round_values=False,covariates=None,fill_na=None,regress_out_method='linear'):
    
    """
    Performs recursive feature elimination (RFE) to select the best subset of features based on a 
    scoring metric. Iteratively removes features that lead to the smallest decrease in the scoring metric.

    Parameters
    ----------
    model : object
        Model instance to train and evaluate on the feature subsets.
    X : pd.DataFrame
        Feature dataset for model training and evaluation.
    y : pd.Series or np.array
        Target variable for training and validation.
    iterator : cross-validation generator
        Cross-validation iterator to split the data into training and validation sets.
    scoring : str, optional
        Scoring metric used to evaluate feature subsets (e.g., 'roc_auc_score') (default is 'roc_auc_score').
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').
    cmatrix : CostMatrix, optional
        Cost matrix for classification, defaults to None.
    priors : dict, optional
        Class priors for probability calibration in classification.
    threshold : float, optional
        Decision threshold for classification tasks.

    Returns
    -------
    best_features : list
        List of selected features after recursive feature elimination.
        
    """

    features = list(X.columns)
    
    # Ascending if error, loss, or other metrics where lower is better
    ascending = any(x in scoring for x in ['error', 'loss', 'cost', 'entropy'])

    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    outputs = np.full((n_samples, n_classes),fill_value=np.nan) if problem_type == 'clf' else np.full(n_samples,fill_value=np.nan)
    y_pred = np.full(n_samples,fill_value=np.nan)
    y_true = np.full(n_samples,fill_value=np.nan)

    for train_index, val_index in iterator.split(X, y, groups):
        X_train = X.loc[train_index]
        X_val = X.loc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        if covariates is not None:
            covariates_train = covariates.loc[train_index].reset_index(drop=True)
            covariates_val = covariates.loc[val_index].reset_index(drop=True)
        else:
            covariates_train, covariates_val = None, None

        model.train(X_train, y_train, covariates_train,fill_na,regress_out_method)
        
        if problem_type == 'clf':
            outputs[val_index] = model.eval(X_val,problem_type,covariates_val,fill_na)
            if isinstance(threshold,float) & (n_classes == 2):
                y_pred[val_index] = [1 if x > threshold else 0 for x in outputs[val_index,1]]
            else:
                y_pred[val_index] = bayes_decisions(scores=outputs[val_index],costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
        else:
            outputs[val_index] = model.eval(X_val,problem_type,covariates_val,fill_na)
            y_pred[val_index] = np.round(outputs[val_index],decimals=0) if round_values else outputs[val_index]
        y_true[val_index] = y_val
    
    outputs = outputs[~np.isnan(y_true)]
    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]

    if scoring == 'roc_auc':
        best_score = roc_auc_score(y_true, outputs[:, 1])
    elif scoring == 'norm_expected_cost':
        best_score = average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
    elif scoring == 'norm_cross_entropy':
        best_score = LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy() if priors is not None else -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy()
    elif 'error' in scoring:
        best_score = eval(scoring)(y_true, y_pred)
    else:
        best_score = eval(f"{scoring}_score")(y_true, y_pred)

    best_features = features.copy()

    while len(features) > 1:
        scorings = {}  # Dictionary to hold scores for each feature removal
        
        for feature in features:
            outputs = np.full((n_samples, n_classes),fill_value=np.nan) if problem_type == 'clf' else np.full(n_samples,fill_value=np.nan)
            y_pred = np.full(n_samples,fill_value=np.nan)
            y_true = np.full(n_samples,fill_value=np.nan)
            
            for train_index, val_index in iterator.split(X, y, groups):
                X_train = X.iloc[train_index][[f for f in features if f != feature]]
                X_val = X.iloc[val_index][[f for f in features if f != feature]]
                y_train, y_val = y[train_index], y[val_index]
                if covariates is not None:
                    covariates_train = covariates.iloc[train_index].reset_index(drop=True)
                    covariates_val = covariates.iloc[val_index].reset_index(drop=True)
                else:
                    covariates_train = None
                model.train(X_train, y_train, covariates_train,fill_na,regress_out_method)
                
                if problem_type == 'clf':
                    outputs[val_index] = model.eval(X_val,problem_type,covariates_val,fill_na)
                    if isinstance(threshold,float) & (n_classes == 2):
                        y_pred[val_index] = [1 if x > threshold else 0 for x in outputs[val_index,1]]
                    else:
                        y_pred[val_index] = bayes_decisions(scores=outputs[val_index],costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
                else:
                    outputs[val_index] = model.eval(X_val,problem_type,covariates_val,fill_na)
                    y_pred[val_index] = np.round(outputs[val_index],decimals=0) if round_values else outputs[val_index]
                y_true[val_index] = y_val
            
            outputs = outputs[~np.isnan(y_true)]
            y_pred = y_pred[~np.isnan(y_true)]
            y_true = y_true[~np.isnan(y_true)]

            if scoring == 'roc_auc':
                scorings[feature] = roc_auc_score(y_true, outputs[:, 1])
            elif scoring == 'norm_expected_cost':
                scorings[feature] = average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
            elif scoring == 'norm_cross_entropy':
                scorings[feature] = LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy() if priors is not None else -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy()
            elif 'error' in scoring:
                scorings[feature] = eval(scoring)(y_true, y_pred)
            else:
                scorings[feature] = eval(f"{scoring}_score")(y_true, y_pred)

        # Sort features by score to find the best to remove
        scorings = pd.DataFrame(list(scorings.items()), columns=['feature', 'score']).sort_values(
            by='score', ascending=ascending).reset_index(drop=True)
        
        best_feature_score = scorings['score'][0]
        feature_to_remove = scorings['feature'][0]
        
        # If improvement is found, update best score and feature set
        if new_best(best_score, best_feature_score, not ascending):
            best_score = best_feature_score
            features.remove(feature_to_remove)
            best_features = features.copy()
            print(f'Removing feature: {feature_to_remove}. New best score: {best_score}')
        else:
            print('No improvement found. Stopping RFE.')
            # Stop if no improvement
            break

    return best_features, best_score

def new_best(old,new,greater=True):
    if greater:
        return new > old
    else:
        return new < old

def tuning(model,scaler,imputer,X,y,groups,hyperp_space,iterator,init_points=5,n_iter=50,scoring='roc_auc',problem_type='clf',cmatrix=None,priors=None,threshold=None,random_state=42,calmethod=None,calparams=None,round_values=False,covariates=None,fill_na=None,regress_out_method='linear'):
    
    def objective(**params):
        return scoring_bo(params, model, scaler, imputer, X, y, groups, iterator, scoring, problem_type, 
                          cmatrix, priors, threshold,calmethod,calparams,round_values,covariates,fill_na,regress_out_method=regress_out_method)
    
    search = BayesianOptimization(f=objective,pbounds=hyperp_space,verbose=2,random_state=random_state)
    #search = BayesSearchCV(model(),hyperp_space,scoring=lambda params,X,y: scoring_bo(params,model,scaler,imputer,X,y,iterator,scoring,problem_type,cmatrix,priors,threshold),n_iter=50,cv=None,random_state=42,verbose=2)
    search.maximize(init_points=init_points,n_iter=n_iter)
    #search.fit(X,y)
    best_params = search.max['params']

    int_params = ['n_estimators', 'n_neighbors', 'max_depth']
    for param in int_params:
        if param in best_params:
            best_params[param] = int(best_params[param])
    return best_params, search.max['target']

def scoring_bo(params,model_class,scaler,imputer,X,y,groups,iterator,scoring,problem_type,cmatrix=None,priors=None,threshold=None,calmethod=None,calparams=None,round_values=False,covariates=None,fill_na=None,regress_out_method='linear'):

    """
    Evaluates a model's performance using cross-validation and a specified scoring metric, 
    facilitating hyperparameter optimization with Bayesian optimization or similar approaches.

    Parameters
    ----------
    params : dict
        Dictionary of model hyperparameters.
    model_class : class
        The model class to instantiate for training and evaluation.
    scaler : callable
        Function to initialize and fit a scaler for feature preprocessing.
    imputer : callable
        Function to initialize and fit an imputer for missing value handling.
    X : pd.DataFrame
        Feature data for training and evaluation.
    y : pd.Series or np.array
        Target variable.
    iterator : cross-validation generator
        Cross-validation iterator to split the data into training and testing sets.
    scoring : str
        Scoring metric for evaluating model performance (e.g., 'roc_auc_score').
    problem_type : str
        Specifies 'clf' for classification or 'reg' for regression tasks.
    cmatrix : CostMatrix, optional
        Cost matrix for classification, defaults to None.
    priors : dict, optional
        Class priors for probability calibration in classification tasks.
    threshold : float, optional
        Decision threshold for classification predictions.

    Returns
    -------
    float
        The computed score based on the chosen scoring metric and the model's cross-validated performance.
    """

    if 'n_estimators' in params.keys():
        params['n_estimators'] = int(params['n_estimators'])
    elif 'n_neighbors' in params.keys():
        params['n_neighbors'] = int(params['n_neighbors'])
    elif 'max_depth' in params.keys():
        params['max_depth'] = int(params['max_depth'])
    if 'random_state' in params.keys():
        params['random_state'] = int(42)
    
    if hasattr(model_class(),'probability') and problem_type == 'clf':
        params['probability'] = True
    
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))

    y_true = np.full(n_samples,fill_value=np.nan)
    y_pred = np.full(n_samples,fill_value=np.nan)
    outputs = np.full((n_samples,n_classes),fill_value=np.nan) if problem_type == 'clf' else np.full(n_samples,fill_value=np.nan)
    
    if isinstance(y,pd.Series):
        y = y.values

    for train_index, test_index in iterator.split(X,y,groups):
        model = Model(model_class(**params),scaler,imputer,calmethod,calparams)
        if covariates is not None:
            covariates_train = covariates.loc[train_index].reset_index(drop=True)
            covariates_val = covariates.loc[test_index].reset_index(drop=True)
        else:
            covariates_train, covariates_val = None, None
        model.train(X.loc[train_index].reset_index(drop=True),y[train_index],covariates_train,fill_na,regress_out_method)
        outputs[test_index] = model.eval(X.loc[test_index].reset_index(drop=True),problem_type,covariates_val,fill_na)

        if problem_type == 'clf':
            if isinstance(threshold,float) & (n_classes == 2):
                y_pred[test_index] = [1 if x > threshold else 0 for x in outputs[test_index,1]]
            else:
                y_pred[test_index] = bayes_decisions(scores=outputs[test_index],costs=cmatrix,priors=priors,score_type='log_posteriors')[0]
        else:
            y_pred[test_index] = np.round(outputs[test_index],decimals=0) if round_values else outputs[test_index]
        y_true[test_index] = y[test_index]
    
    y_true = y_true[~np.isnan(y_true)]
    y_pred = y_pred[~np.isnan(y_pred)]
    outputs = outputs[~np.isnan(outputs).all(axis=1)] if problem_type == 'clf' else outputs[~np.isnan(outputs)]

    if 'error' in scoring:
        return -eval(scoring)(y_true, outputs if problem_type == 'reg' else y_pred)
    elif scoring == 'norm_expected_cost':
        return -average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
    elif scoring == 'norm_cross_entropy':
        return -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy() if priors is not None else -LogLoss(log_probs=torch.tensor(outputs),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy()
    elif scoring == 'roc_auc':
        return roc_auc_score(y_true, outputs[:,1])
    else:
        return eval(f"{scoring}_score")(y_true, y_pred)

def compare(models,X_dev,y_dev,iterator,random_seeds_train,metric_name,IDs_dev,n_boot=100,cmatrix=None,priors=None,problem_type='clf'):
    metrics = np.empty((np.max((1,n_boot)),len(models)))
    
    for m,model in enumerate(models.keys()):
        _,metrics_bootstrap,_,_,_,_,_ = CV(0,models[model],X_dev[model],y_dev,X_dev[model].columns,iterator,random_seeds_train,metric_name,IDs_dev,n_boot=n_boot,cmatrix=cmatrix,priors=priors,problem_type=problem_type)
        metrics[:,m] = metrics_bootstrap[metric_name[0]]
    return metrics

def css(metrics,scoring='roc_auc',problem_type='clf'):
    inf_conf_int = np.empty(metrics[scoring].shape[0])
    sup_conf_int = np.empty(metrics[scoring].shape[0])

    for model in range(metrics[scoring].shape[0]):
        _, inf_conf_int[model], sup_conf_int[model] = conf_int_95(metrics[scoring][model])
        
    if problem_type == 'clf':
        if 'norm' not in scoring:
            best = np.argmax(inf_conf_int)
        else:
            best = np.argmin(sup_conf_int)
    else:
        if 'error' in scoring:
            best = np.argmin(sup_conf_int)
        else:
            best = np.argmax(inf_conf_int)
            
    return best

def select_best_models(metrics,scoring='roc_auc',problem_type='clf'):

    best = css(metrics,scoring,problem_type)
    return best