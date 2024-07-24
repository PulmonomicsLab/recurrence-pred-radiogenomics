from typing import Tuple
from sklearn.metrics import roc_curve, auc
from scipy import interp
from .model import Model
import numpy as np


def plot_roc_model_cv(model: Model, verbose: bool = False) -> Tuple[np.array, np.array, float, float]:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(model.estimators)):
        x_test = model.data[model.test_indices[i]]
        y_test = model.target[model.test_indices[i]]
        probas = model.get_decision_score(model.estimators[i], x_test)
        if model.estimator_id == 'SVM':
            fpr, tpr, thresholds = roc_curve(y_test, probas)
        elif model.estimator_id == 'RF':
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        elif model.estimator_id == 'MLP':
            fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('AUC(CV ' + str(i) + ') = ' + str(roc_auc))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = float(np.std(aucs))

    if verbose:
        print('\tMean AUC = ' + str(mean_auc) + ' (+/- ' + str(std_auc) + ')')

    # ------ Print as CSV ------#
    # heading = ['FPR']
    # for i in range(len(model.estimators)):
    #     heading.append('TPR' + str(i) + ' AUC = ' + str(round(aucs[i], 4)))
    # heading.append('Mean TPR' + ' AUC = ' + str(round(mean_auc, 4)) + ' (+/- ' + str(round(std_auc, 4)) + ')')
    # print('\t'.join(heading))
    # for i in range(100):
    #     row = [mean_fpr[i]]
    #     for j in range(len(model.estimators)):
    #         row.append(tprs[j][i])
    #     row.append(mean_tpr[i])
    #     print('\t'.join(map(str, row)))
    # ------ Print as CSV ------#

    return mean_fpr, mean_tpr, mean_auc, std_auc


def plot_roc_model_blind(model: Model, b_data: np.array, b_target: np.array, verbose: bool = False) \
        -> Tuple[np.array, np.array, float]:
    mean_fpr = np.linspace(0, 1, 100)
    probas = model.get_decision_score(model.total_estimator, b_data)
    if model.estimator_id == 'SVM':
        fpr, tpr, thresholds = roc_curve(b_target, probas)
    elif model.estimator_id == 'RF':
        fpr, tpr, thresholds = roc_curve(b_target, probas[:, 1])
    elif model.estimator_id == 'MLP':
        fpr, tpr, thresholds = roc_curve(b_target, probas[:, 1])
    mean_tpr = interp(mean_fpr, fpr, tpr)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    if verbose:
        print('\tMean AUC = ' + str(mean_auc))

    # ------ Print as CSV ------#
    # heading = ['FPR', 'Mean TPR' + ' AUC = ' + str(round(mean_auc, 4))]
    # print('\t'.join(heading))
    # for i in range(100):
    #     row = [str(mean_fpr[i]), str(mean_tpr[i])]
    #     print('\t'.join(row))
    # ------ Print as CSV ------#

    return mean_fpr, mean_tpr, mean_auc
