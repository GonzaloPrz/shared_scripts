import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
import pandas as pd
from sklearn.svm import SVR, SVC
from pathlib import Path

from expected_cost.ec import *
from expected_cost.utils import *
from expected_cost.calibration import *

from psrcal.losses import LogLoss
from psrcal.calibration import *

import shap, pickle

class Model():
    def __init__(self,model,scaler=None,imputer=None,calmethod=None,calparams=None):
        self.model = model
        self.scaler = scaler() if scaler is not None else None
        self.imputer = imputer() if imputer is not None else None
        self.calmethod = calmethod
        self.calparams = calparams
        self.regress_out_model = None

    def train(self,X,y,covariates=None,fill_na=None,regress_out_method='linear'):   
        
        X_t = self.scaler.fit_transform(X.values) if self.scaler is not None else X.values
        
        if fill_na is not None:
            nan_mask = np.isnan(X_t)
            X_t[nan_mask] = fill_na
        else:
            X_t = self.imputer.fit_transform(X_t) if self.imputer is not None else X_t

        X_t = pd.DataFrame(data=X_t,columns=X.columns)
        params = self.model.get_params()
        if 'n_estimators' in params.keys():
            params['n_estimators'] = int(params['n_estimators']) if params['n_estimators'] is not None else None
        if 'n_neighbors' in params.keys():
            params['n_neighbors'] = int(params['n_neighbors'])
        if 'max_depth' in params.keys():
            params['max_depth'] = int(params['max_depth']) if params['max_depth'] is not None else None
        if 'max_iter' in params.keys():
            params['max_iter'] = int(params['max_iter']) if params['max_iter'] is not None else None
        if 'gpu_id' in params.keys():
            params['gpu_id'] = None
        if 'min_samples_leaf' in params.keys():
            params['min_samples_leaf'] = int(params['min_samples_leaf'])
                
        self.model.set_params(**params)
        if hasattr(self.model,'precompute'):
            self.model.precompute = True
        
        if covariates is not None:
            
            self.regress_out_model = dict((feature,None) for feature in X_t.columns)

            for feature in X_t.columns:
                feature_, model = regress_out_fn(data=pd.concat((X_t,covariates),axis=1),target_column=feature,covariate_columns=covariates.columns,method=regress_out_method)
                X_t[feature] = feature_
                self.regress_out_model[feature] = model

        self.model.fit(X_t,y)
        
    def eval(self,X,problem_type='clf',covariates=None,fill_na=None):
        
        X_input = X.values if hasattr(X,'values') else X
        X_t = self.scaler.transform(X_input) if self.scaler is not None else X_input
        
        if fill_na is not None:
            nan_mask = np.isnan(X_t)
            X_t[nan_mask] = fill_na
        else:
            X_t = self.imputer.transform(X_t) if self.imputer is not None else X_t
        
        X_t = pd.DataFrame(data=X_t,columns=X.columns)

        if covariates is not None:
                
            for feature in X_t.columns:
                X_t[feature] = X_t[feature] - self.regress_out_model[feature].predict(covariates)

        if problem_type == 'clf':
            prob = self.model.predict_proba(X_t)
            prob = np.clip(prob,1e-2,1-1e-2)
            score = np.log(prob)
        else:
            score = self.model.predict(X_t)

        score_filled = score.copy()

        if problem_type == 'clf':
            nan_indices_col0 = np.isnan(score[:, 0]) # True where first column is NaN
            nan_indices_col1 = np.isnan(score[:, 1])  # True where second column is NaN

            # Replace them accordingly:
            score_filled[nan_indices_col0, 0] = np.log(1e-2)
            score_filled[nan_indices_col1, 1] =  np.log(1-1e-2)
            
        return score_filled
    
    def calibrate(self,logpost_tst,targets_tst,logpost_trn=None,targets_trn=None):
        
        if logpost_trn is None:
            cal_outputs_test = calibration_with_crossval(logpost=logpost_tst,targets=targets_tst,calmethod=self.calmethod,calparams=self.calparams)        
            calmodel = None
        else:
            cal_outputs_test, calmodel = calibration_train_on_heldout(logpost_trn=logpost_trn,targets_trn=targets_trn,logpost_tst=logpost_tst,calmethod=self.calmethod,calparams=self.calparams,return_model=True)        

        return cal_outputs_test, calmodel

def get_base_dir(project_name):
    base_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name) 

    return base_dir

def _build_path(base_dir, task, dimension, y_label, random_seed_test, file_name, config, bayes=False, scoring=None):
    """Constructs a standardized file path from configuration options."""
    hyp_opt_str = "hyp_opt" if config["n_iter"] else ""
    feature_sel_str = "feature_selection" if bool(config['feature_selection']) else ""
    outlier_str = "filter_outliers" if config['filter_outliers'] and config["problem_type"] == 'reg' else ''
    round_str = "rounded" if config["round_values"] else ""
    cut_str = "cut" if config["cut_values"] > 0 else ""
    shuffle_str = "shuffle" if config["shuffle_labels"] else ""

    path = Path(base_dir,task, dimension, config['kfold_folder'], 
           y_label, config["stat_folder"], scoring if bayes else '', hyp_opt_str, feature_sel_str, outlier_str, round_str, cut_str,shuffle_str, random_seed_test, config['version'], file_name)

    if not path.exists():
        path = Path(str(path).replace(config['version'],''))
    if not path.exists():
        path = Path(str(path).replace('.npy','.pkl'))
        
    return path

def _load_data(results_dir, task, dimension, y_label, model_type, random_seed_test, config, bayes=False,scoring=None):
    """Loads model outputs and true labels for a given configuration."""
    path_kwargs = {'base_dir': results_dir,'task': task, 'dimension': dimension, 'y_label': y_label, 'random_seed_test': random_seed_test}

    if 'round_values' not in config:
        config["round_values"] = False
    
    if "cut_values" not in config:
        config["cut_values"] = 0

    IDs_path = _build_path(**path_kwargs, file_name=f'IDs_dev.npy',config=config,bayes=bayes,scoring=scoring)    
    outputs_path = _build_path(**path_kwargs,file_name=f'outputs_{model_type}_calibrated.npy' if config["calibrate"] else f'outputs_{model_type}.npy',config=config,bayes=bayes,scoring=scoring)
    y_dev_path = _build_path(**path_kwargs, file_name=f'y_dev.npy',config=config,bayes=bayes,scoring=scoring)

    with open(IDs_path, "rb") as f:
        # Load and select the specific model's outputs
        try:
            IDs_dev = np.load(f,allow_pickle=True)
        except:
            IDs_dev = pickle.load(f)

    with open(outputs_path, "rb") as f:
        # Load and select the specific model's outputs
        try:
            outputs_dev = np.load(f,allow_pickle=True)
        except:
            outputs_dev = pickle.load(f)

    with open(y_dev_path, "rb") as f:

        try:
            y_dev = np.load(f,allow_pickle=True)
        except:
            y_dev = pickle.load(f)

    IDs_path = _build_path(**path_kwargs, file_name=f'IDs_test.npy',config=config,bayes=bayes,scoring=scoring)    
    outputs_path = _build_path(**path_kwargs,file_name=f'outputs_test_{model_type}_calibrated.npy' if config["calibrate"] else f'outputs_test_{model_type}.npy',config=config,bayes=bayes,scoring=scoring)
    y_test_path = _build_path(**path_kwargs, file_name=f'y_test.npy',config=config,bayes=bayes,scoring=scoring)

    try:
        with open(IDs_path, "rb") as f:
            # Load and select the specific model's outputs
            try:
                IDs_test = np.load(f,allow_pickle=True)
            except:
                IDs_test = pickle.load(f)

        with open(outputs_path, "rb") as f:
            # Load and select the specific model's outputs
            try:
                outputs_test = np.load(f,allow_pickle=True)
            except:
                outputs_test = pickle.load(f)

        with open(y_test_path, "rb") as f:
            try:
                y_test = np.load(f,allow_pickle=True)
            except:
                y_test = pickle.load(f)
    except:
        IDs_test, outputs_test, y_test = None, None, None

    return IDs_dev, outputs_dev, y_dev, IDs_test, outputs_test, y_test

def filter_outliers(data,parametric=True,n_sd=3):
    for feature in data.columns:
        if parametric:
            data = data[np.abs(data[feature]-np.nanmean(data[feature]))/np.nanstd(data[feature]) < n_sd]
        else:
            data = data[np.abs(data[feature] - np.nanmedian(data[feature]))/(np.abs(np.nanpercentile(data[feature],q=75) - np.nanpercentile(data[feature],q=25))) < 1.5]
    
    return data

def regress_out_fn(
    data: pd.DataFrame,
    target_column: str,
    covariate_columns: list[str],
    method = 'linear'
):
    """
    Ajusta y = beta0 + betaX y devuelve los residuos (y - y_hat).
    """
    covariate_matrix = data[covariate_columns]
    target_vector = data[target_column]
    model = LinearRegression() if method == 'linear' else RandomForestRegressor(random_state=42)
    
    model.fit(covariate_matrix,target_vector)

    fitted_values = model.predict(covariate_matrix)
    residuals = target_vector - fitted_values

    return pd.Series(
        residuals,
        index=data.index,
        name=f"{target_column}_residual",
    ), model

def run_shap_analysis(model_wrapper, X_dev, y_dev, groups, iterator,fill_na=0):
    """
    Genera gráficos SHAP agnósticos al modelo (Linear, Tree, o Kernel).
    
    Args:
        model_wrapper: Tu objeto clase Model (de utils.py)
        X_train: Datos usados para entrenar (para fondo de referencia)
        X_test: Datos a explicar (el set de test)
        feature_names: Lista de nombres de las columnas
        output_dir: Ruta donde guardar las imágenes
    """
    
    # 1. Extraer el modelo subyacente de tu clase wrapper
    #model = model_wrapper.model
    
    model_type = type(model_wrapper.model).__name__
    print(f"Calculando SHAP para modelo tipo: {model_type}")

    explainer = None
    shap_values = pd.DataFrame(columns=X_dev.columns,index=range(X_dev.shape[0]),dtype=float)
    
    try:
        for train_index, val_index in iterator.split(X_dev,y_dev,groups):
            X_train = X_dev.loc[train_index]
            X_val = X_dev.loc[val_index]

            # A. Modelos basados en Árboles (XGBoost, Random Forest) -> TreeExplainer (Rápido y Exacto)
            if 'XGB' in model_type or 'Forest' in model_type or 'Tree' in model_type:
                explainer = shap.TreeExplainer(model_wrapper.model)
                shap_values.loc[val_index] = explainer.shap_values(X_val)
                
                # Fix para XGBoost binario que a veces devuelve lista
                if isinstance(shap_values, list):
                    shap_values.loc[val_index] = shap_values[1] 

            # B. Modelos Lineales (LogisticRegression, Ridge, Lasso, SVM-Linear) -> LinearExplainer
            elif 'Linear' in model_type or 'Logistic' in model_type or 'Ridge' in model_type or 'Lasso' in model_type:
                # LinearExplainer necesita un "background" para comparar (usamos X_train resumen)
                masker = shap.maskers.Independent(X_train)
                explainer = shap.LinearExplainer(model_wrapper.model, masker)
                shap_values.loc[val_index] = explainer.shap_values(X_val)

            # C. Modelos Kernel/Caja Negra (SVM-RBF, KNN) -> KernelExplainer (Lento pero universal)
            else:
                print("Usando KernelExplainer (esto puede tardar)...")
                # Usamos un resumen del train set porque KernelExplainer es computacionalmente costoso
                X_train = pd.DataFrame(columns=X_train.columns,data=model_wrapper.imputer.transform(X_train.values))
               
                if fill_na == 0:
                    X_train = pd.DataFrame(columns=X_train.columns,data=model_wrapper.scaler.transform(X_train.values))
                else:
                    nan_mask = np.isnan(X_train.values)
                    X_train.loc[nan_mask] = fill_na

                background = shap.kmeans(X_train, np.min((50,X_train.shape[0]))) 
                
                # Nota: KernelExplainer necesita la función de predicción de probabilidad si es clf
                
                X_val = pd.DataFrame(columns=X_val.columns,data=model_wrapper.scaler.transform(X_val.values))
                if fill_na == 0:
                    X_val = pd.DataFrame(columns=X_val.columns,data=model_wrapper.imputer.transform(X_val.values))
                else:
                    nan_mask = np.isnan(X_val.values)
                    X_val.loc[nan_mask] = fill_na

                if hasattr(model_wrapper.model, 'predict_proba'):
                    predict_fn = lambda x: model_wrapper.model.predict_proba(x)[:, 1] # Probabilidad clase 1
                else:
                    predict_fn = model_wrapper.model.predict  

                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values.loc[val_index] = explainer.shap_values(X_val)

        return shap_values
    except Exception as e:
        print(f"Error al calcular SHAP: {e}")
        print("Sugerencia: Revisa si el modelo soporta introspección directa o si hay mismatch de dimensiones.")
        return None