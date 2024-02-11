import numpy as np


def binary_classification_metrics(y_true, y_pred):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = ((y_pred == 1) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    
    #When calculating accuracy, 0/0 may occur if no objects were submitted an empty array for prediction (it was checked in knn.py)
    accuracy = (TN + TP) / (TN + TP + FP + FN)
    
    #When calculating precision, 0/0 may occur in 2 cases: 
    #1) when the model always predicts Negative (TP=0, FP =0), then it is logical to put 0; 
    #2) when there was no positive objects at all in the test and they were not predicted, then it seems right to put 1 
    #(since all test objects (<=> all zeros) were predicted correctly (<=> all zeros)). We'll implement it:
    if (np.sum((y_true == 0)) == y_true.shape) & (np.sum((y_pred == 0)) == y_pred.shape):
        precision = 1 #it might be usefull to call a warning that there is only 0 in the test here.
    elif TP == 0 & FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    
    #When calculating recall, 0/0 may occur in the following case: 
    #when there was no positive at all in the test (=> TP = 0), 
    #then the model could not predict False Negative (=> FN = 0). In this case, we will set 1.
    if (np.sum((y_true == 0)) == y_true.shape):
        recall = 1 #it might be usefull to call a warning that there is only 0 in the test here.
    else:
        recall = TP / (TP + FN)   
    
    #When calculating f1-score, 0/0 may occur in the following case: 
    #when there was no positive at all in the test (=> TP = 0) 
    #and then the model could not predict False Negative (=> FN = 0),
    #but it did not predict positive also (FP = 0)
    if (np.sum((y_true == 0)) == y_true.shape) & (np.sum((y_pred == 0)) == y_pred.shape):
        f1 = 1 #it might be usefull to call a warning that there is only 0 in the test here.
    else:
        f1 = 2*TP/ (2*TP + FP + FN) # 2 * (precision * recall) / (precision + recall) 
    return accuracy, precision, recall, f1


def multiclass_accuracy(y_true, y_pred):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    accuracy = (y_pred == y_true).sum()/y_true.shape[0]
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    n = y_true.shape[0]
    y_av = np.mean(y_true)
    r2 = 1.0 - (np.sum(np.power((y_true - y_pred), 2))) / (np.sum(np.power((y_true - y_av), 2)))
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    n = y_true.shape[0]
    mse = np.sum(np.power((y_true - y_pred), 2)) / n
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    n = y_true.shape[0]
    mae = np.sum(np.abs(y_true - y_pred)) / n
    return mae
    