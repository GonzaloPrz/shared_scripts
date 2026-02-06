import numpy as np

from expected_cost.ec import *
from expected_cost.utils import *

from sklearn.metrics import roc_auc_score, log_loss, average_precision_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score

from psrcal.losses import LogLoss
from psrcal.calibration import *

from scipy.stats import bootstrap

import itertools

def _calculate_metrics(indices, outputs, y, metrics_names, prob_type, cost_matrix):
    """
    Statistic function for bootstrap. Calculates differences for ALL metrics at once.
    """
    # Resample y, ensuring we don't operate on an empty or invalid slice
    while y.ndim < 3:
        y = y[np.newaxis,:]
    
    resampled_y = y[:, :, indices].ravel()

    # If a resample is degenerate (e.g., missing a class), metric calculation is impossible.
    # Return NaNs to signal this. The 'bca' method will fail, triggering our fallback.
    if prob_type == 'clf':
        while np.unique(resampled_y).shape[0] != np.unique(y).shape[0]:
            np.random.seed(np.random.randint(0,1e6))
            indices = np.random.choice(np.arange(len(indices)),len(indices),replace=True)
            resampled_y = y[:, :, indices].ravel()

    # Resample model outputs

    while ((prob_type == 'clf') & (outputs.ndim < 4)) | ((prob_type == 'reg') & (outputs.ndim < 3)):
        outputs = outputs[np.newaxis,:]

    resampled_out = outputs[:, :, indices].reshape(-1, outputs.shape[-1]) if prob_type == 'clf' else outputs[:,:,indices].ravel()

    # Get metrics for both classifiers
    if prob_type == 'clf':
        metrics, _ = get_metrics_clf(resampled_out, resampled_y, metrics_names, cmatrix=cost_matrix)
    else: # 'reg'
        metrics = get_metrics_reg(resampled_out, resampled_y, metrics_names)

    # Return an array of differences
    return np.array([metrics[m] for m in metrics_names])

def get_metrics_clf(y_scores,y_true,metrics_names,cmatrix=None,priors=None,threshold=None,weights=None):
    """
    Calculates evaluation metrics for the predicted scores and true labels.

    Args:
        y_scores: The predicted scores.
        y_true: The true labels.
        metrics_names: The names of the evaluation metrics.
        cmatrix: The cost matrix used to calculate expected costs. Defaults to None.
        priors: The prior probabilities of the target classes. Defaults to None.

    Returns:
        metrics: A dictionary containing the evaluation metrics.

    Note:
        The function calculates the evaluation metrics for the predicted scores and true labels.
    """
    try:
        if np.isnan(threshold):
            threshold = None
    except:
        threshold = None

    if cmatrix is None:
        cmatrix = CostMatrix.zero_one_costs(K=y_scores.shape[-1])

    y_pred = bayes_decisions(scores=y_scores,costs=cmatrix,priors=priors,score_type='log_posteriors')[0] if threshold is None else np.array(y_scores[:,1] > threshold,dtype=int)

    metrics = dict([(metric,[]) for metric in metrics_names])

    for m in metrics_names:
        if m == 'norm_cross_entropy':
            metrics[m] = float(LogLoss(log_probs=torch.tensor(y_scores),labels=torch.tensor(np.array(y_true),dtype=torch.int),priors=torch.tensor(priors)).detach().numpy()) if priors is not None else float(LogLoss(log_probs=torch.tensor(y_scores),labels=torch.tensor(np.array(y_true),dtype=torch.int)).detach().numpy())
        elif m == 'norm_expected_cost':
            metrics[m] = average_cost(targets=np.array(y_true,dtype=int),decisions=np.array(y_pred,dtype=int),costs=cmatrix,priors=priors,adjusted=True)
        elif m == 'roc_auc':
            metrics[m] = roc_auc_score(y_true=y_true,y_score=y_scores[:,1],sample_weight=weights)
        else:
            metrics[m] = eval(f'{m}_score')(y_true=np.array(y_true,dtype=int),y_pred=y_pred,sample_weight=weights)
        
    return metrics,y_pred

def get_metrics_reg(y_scores,y_true,metrics_names):
    """
    Calculates evaluation metrics for the predicted scores and true labels.

    Args:
        y_scores: The predicted scores.
        y_true: The true labels.
        metrics_names: The names of the evaluation metrics.

    Returns:
        metrics: A dictionary containing the evaluation metrics.

    Note:
        The function calculates the evaluation metrics for the predicted scores and true labels.
    """
    metrics = dict((metric,np.nan) for metric in metrics_names)	
    for metric in metrics_names:

        try:
            metrics[metric] = eval(metric)(y_true=y_true,y_pred=y_scores)
        except:
            metrics[metric] = eval(f'{metric}_score')(y_true=y_true,y_pred=y_scores)

    return metrics

def conf_int_95(data):
    mean = np.nanmean(data)
    inf = np.nanpercentile(data,2.5)
    sup = np.nanpercentile(data,97.5) 
    return mean, inf, sup

def compute_metrics(model_index, outputs, y_dev, metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=None):
    # Calculate the metrics using the bootstrap method
    if outputs.ndim == 4 and problem_type == 'clf':
        outputs = outputs[:,np.newaxis,:,:,:]
    elif outputs.ndim == 3 and problem_type == 'reg':
        outputs = outputs[:,np.newaxis,:,:]
    if outputs.shape[-1] > 2:
        metrics_names = list(set(metrics_names) - set(['roc_auc','f1','recall','precision']))
    
    if cmatrix is None:
        cmatrix = CostMatrix.zero_one_costs(K=outputs.shape[-1])

    n_samples = y_dev.shape[0]

    data = (np.arange(n_samples),)

    ci_metrics = dict((metric,[]) for metric in metrics_names)

    def metric_func(metric, indices):
        metric_result = []

        for j,r in itertools.product(range(y_dev.shape[0]),range(y_dev.shape[1])):
            yt = y_dev[j,r,indices].squeeze()
            if problem_type == 'clf':
                if threshold is not None and np.unique(y_dev).shape[0] == 2:
                    yp = (outputs[j,model_index,r,indices,1].squeeze() > threshold).astype(int)
                else:
                    yp = bayes_decisions(scores=outputs[j,model_index,r,indices].squeeze(), costs=cmatrix, priors=priors, score_type='log_posteriors')[0]
            else:
                yp = outputs[j,model_index,r,indices].squeeze()
            ys = outputs[j,model_index,r,indices].squeeze()

            try:
                if metric == 'roc_auc':

                    metric_result += [roc_auc_score(yt,ys[:,1])]
                elif metric == 'norm_cross_entropy':
                    metric_result += [LogLoss(log_probs=torch.tensor(ys), labels=torch.tensor(np.array(yt, dtype=int)), priors=torch.tensor(priors)).detach().numpy() if priors is not None else LogLoss(log_probs=torch.tensor(ys), labels=torch.tensor(np.array(yt, dtype=int))).detach().numpy()]
                elif metric == 'norm_expected_cost':
                    metric_result += [average_cost(targets=np.array(yt, dtype=int), decisions=np.array(yp, dtype=int), costs=cmatrix, priors=priors, adjusted=True)]
                elif 'error' in metric:
                    metric_result += [eval(metric)(yt, yp)]
                else:
                    metric_result += [eval(f'{metric}_score')(yt, yp)]
            except ValueError as e:
                print(f"Error calculating {metric} for indices {indices}: {e}")
                metric_result += [np.nan]    

        return metric_result
      
    for metric in metrics_names:
        #Calculate bootstrap BCa confidence intervals
        res = bootstrap(
            data, 
            lambda idx: metric_func(metric, idx),
            vectorized=False,
            n_resamples=n_boot,
            confidence_level=.95,
            method='bca',
            random_state=42
        )

        estimate = np.round(metric_func(metric,data),3)
        ci = res.confidence_interval
        ci_metrics[metric] = (estimate, (np.round(ci.low,3), np.round(ci.high,3)))

    '''
    #results, sorted_IDs = get_metrics_bootstrap(outputs[j,model_index,r], y_dev[j,r], IDs[j,r],metrics_names, n_boot=n_boot, cmatrix=cmatrix,priors=priors,threshold=threshold,problem_type=problem_type,bayesian=bayesian)

    metrics_result = {}
    for metric in metrics_names:
        metrics_result[metric] = results[metric]
    '''
    return ci_metrics

def get_metrics_bootstrap(samples, targets, IDs, metrics_names, n_boot=2000,cmatrix=None,priors=None,threshold=None,problem_type='clf',bayesian=False):
    
    all_metrics = dict((metric,np.zeros(n_boot)) for metric in metrics_names)
   
    for metric in metrics_names:
        if bayesian:
            weights = np.random.dirichlet(np.ones(samples.shape[0]))
        else:
            weights = None
        #Sort IDs and keep indices to adjust samples and targets' order
        indices_ = np.argsort(IDs)
        samples = samples[indices_]
        targets = targets[indices_]
        sorted_IDs = [IDs[indices_]]
        
        for b in range(n_boot):
            rng = np.random.default_rng(seed=b)
            indices = rng.choice(indices_, len(indices_), replace=True)

            while len(np.unique(targets[indices])) == 1:
                indices = rng.choice(indices_, len(indices_), replace=True)

            if problem_type == 'clf':
                metric_value, y_pred = get_metrics_clf(samples[indices], targets[indices], [metric], cmatrix,priors,threshold,weights)
            else:
                metric_value = get_metrics_reg(samples[indices], targets[indices], [metric])
            if (len(metric_value) == 0) or (not isinstance(metric_value[metric],float)):
                
                continue

            all_metrics[metric][b] = metric_value[metric]
        
    return all_metrics, sorted_IDs
