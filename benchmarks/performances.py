import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve

def performances(y_true, y_pred, y_prob, print_=True):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                      labels=[0, 1]).ravel().tolist()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    try:
        mcc = ((tp * tn) - (fn * fp)) / np.sqrt(
            float((tp + fn) * (tn + fp) * (tp + fp) * (tn + fn)))
    except:
        print('MCC Error: ', (tp + fn) * (tn + fp) * (tp + fp) * (tn + fn))
        mcc = np.nan
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    try:
        recall = tp / (tp + fn)
    except:
        recall = np.nan

    try:
        precision = tp / (tp + fp)
    except:
        precision = np.nan

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = np.nan

    roc_auc = roc_auc_score(y_true, y_prob)
    # roc_auc = roc_auc_score(y_true, y_prob, max_fpr=0.1)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)

    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(
            Counter(y_pred)[0],
            Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(
            Counter(y_true)[0],
            Counter(y_true)[1]))
        print(
            'auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'
            .format(roc_auc, sensitivity, specificity, accuracy, mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(
            precision, recall, f1, aupr))

    return (roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision,
            recall, aupr)

