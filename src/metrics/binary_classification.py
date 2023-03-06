import numpy as np


class ConfusionMatrix:
    def __init__(self, y_true=None, y_pred=None):
        self._confusion_matrix = np.array([[0, 0], [0, 0]])
        if not (y_true is None or y_pred is None):
            self.upd(y_true, y_pred)

    def upd(self, y_true, y_pred):
        for value, prediction in zip(y_true, y_pred):
            self._confusion_matrix[value][prediction] += 1

    def get(self):
        return self._confusion_matrix

    def get_flatten(self):
        return self._confusion_matrix.ravel()


def f1_score(y_true, y_pred):
    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.get_flatten()
    return tp / (tp + (fp + fn) / 2)


def accuracy_score(y_true, y_pred):
    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.get_flatten()
    return (tn + tp) / (tn + fp + fn + tp)


def precision_score(y_true, y_pred):
    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.get_flatten()
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.get_flatten()
    return tp / (fn + tp)


def roc_curve(y_true, y_proba):
    tpr, fpr = [], []
    thresholds = np.linspace(0, 1, 1000)

    for threshold in thresholds:
        y_pred = np.where(y_proba >= threshold, 1, 0)
        confusion_matrix = ConfusionMatrix(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix.get_flatten()
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    return fpr, tpr, thresholds


def auc(x, y):
    s = 0
    for i in range(1, len(x)):
        s += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1])
    return abs(s)
