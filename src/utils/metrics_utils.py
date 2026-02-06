import numpy as np

from expected_cost.ec import *
from expected_cost.utils import *

from sklearn.metrics import roc_auc_score, f1_score, precision_score, balanced_accuracy_score, accuracy_score, recall_score, mean_squared_error, mean_absolute_error, r2_score

from psrcal.losses import LogLoss
from psrcal.calibration import *

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