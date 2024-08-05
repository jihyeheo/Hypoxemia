from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, confusion_matrix
)

import numpy as np

def evaluate_metrics(y_true, y_pred, y_prob):
    """
    Computes various evaluation metrics.

    Parameters:
    y_true (list or array): True binary labels.
    y_pred (list or array): Predicted binary labels.
    y_prob (list or array): Predicted probabilities.

    Returns:
    dict: Dictionary with accuracy, recall, precision, specificity, f1 score, auroc, and auprc.
    """
    
    # Ensure y_true, y_pred, y_prob are 1D arrays
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    y_prob = np.ravel(y_prob)
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate accuracy, recall, precision, f1 score
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_prob)
    
    # Calculate AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall_vals, precision_vals)
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'specificity': specificity,
        'f1_score': f1,
        'auroc': auroc,
        'auprc': auprc
    }