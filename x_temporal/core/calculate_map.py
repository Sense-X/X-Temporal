import numpy as np
from sklearn.metrics import average_precision_score


def calculate_mAP(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    values = []
    for i in range(len(y_pred)):
        values.append(
            average_precision_score(
                y_true[i],
                y_pred[i],
                average='macro'))
    return np.mean(values) * 100
