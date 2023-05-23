import numpy as np


class MulticlassConfusionMatrix:
    def __init__(self, y_true=None, y_pred=None):
        self._confusion_matrix = None
        self.unique_labels = None
        if not (y_true is None or y_pred is None):
            self.upd(np.asarray(y_true), np.asarray(y_pred))

    def upd(self, y_true, y_pred):
        self.unique_labels = np.unique(np.hstack((y_true, y_pred)))
        self._confusion_matrix = np.zeros((self.unique_labels.size, self.unique_labels.size), dtype=int)
        for value, prediction in zip(y_true, y_pred):
            self._confusion_matrix[value][prediction] += 1

    def get(self):
        return self._confusion_matrix

    def get_stat(self, label):
        fp, fn = 0, 0
        for i in range(self.get_size()):
            if i != label:
                fp += self._confusion_matrix[i][label]
                fn += self._confusion_matrix[label][i]
        tp = self._confusion_matrix[label][label]
        tn = np.sum(self._confusion_matrix) - (tp + fp + fn)
        return np.array([tn, fp, fn, tp], dtype=int)

    def get_unique_labels(self):
        return self.unique_labels.copy()

    def get_size(self):
        return self.unique_labels.size


def get_sum_stats(confusion_matrix):
    stats = np.zeros(4)
    for label in confusion_matrix.get_unique_labels():
        stats += confusion_matrix.get_stat(label)
    return stats


def get_average_metrics(confusion_matrix, get_score):
    size = confusion_matrix.get_size()
    sum_score = 0
    for label in confusion_matrix.get_unique_labels():
        sum_score += get_score(*confusion_matrix.get_stat(label))
    return sum_score / size


def score(y_true, y_pred, average, get_score):
    confusion_matrix = MulticlassConfusionMatrix(y_true, y_pred)
    if average == "micro":
        return get_score(*get_sum_stats(confusion_matrix))
    return get_average_metrics(confusion_matrix, get_score)


def get_f1_score(tn, fp, fn, tp):
    return tp / (tp + (fp + fn) / 2)


def f1_score(y_true, y_pred, average="micro"):
    return score(y_true, y_pred, average, get_f1_score)


def get_accuracy_score(tn, fp, fn, tp):
    return (tn + tp) / (tn + fp + fn + tp)


def accuracy_score(y_true, y_pred, average="micro"):
    return score(y_true, y_pred, average, get_accuracy_score)


def get_precision_score(tn, fp, fn, tp):
    return tp / (tp + fp)


def precision_score(y_true, y_pred, average="micro"):
    return score(y_true, y_pred, average, get_precision_score)


def get_recall_score(tn, fp, fn, tp):
    return tp / (fn + tp)


def recall_score(y_true, y_pred, average="micro"):
    return score(y_true, y_pred, average, get_recall_score)
