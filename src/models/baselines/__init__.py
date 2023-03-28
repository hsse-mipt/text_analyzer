import numpy as np
import pandas as pd
import scipy
from math import log
from random import choices
from abc import ABC, abstractmethod
from tqdm import tqdm


class DummyClassifier:
    def __init__(self):
        self.classes = []
        self.weights = []

    @staticmethod
    def get_weights(y):
        occurrences = np.array(y.value_counts().tolist())
        return occurrences / np.sum(occurrences)

    def fit(self, y, weights=None):
        self.classes = np.unique(y)
        self.weights = DummyClassifier.get_weights(y) if weights is None else weights

    def predict(self, X):
        return choices(self.classes, weights=self.weights, k=X.shape[0])


class NaiveBayesClassifier:
    def __init__(self):
        self.class_proba = {}
        self.unique_y = None

    def fit(self, X_train, y_train):
        self.unique_y = np.unique(y_train)
        for y in self.unique_y:
            current_y_feature_matrix = X_train[y_train == y]
            self.class_proba[y] = current_y_feature_matrix.size / X_train.size

    def predict(self, test_feature_matrix):
        y_pred = []
        predict = None
        for row in tqdm(range(test_feature_matrix.shape[0])):
            max_likelihood = -float('inf')
            for y in self.unique_y:
                likelihood = log(self.class_proba[y])
                for x_i in range(test_feature_matrix.shape[1]):
                    likelihood += scipy.stats.norm.logpdf(test_feature_matrix.iloc[row][x_i])
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    predict = y
            y_pred.append(predict)
        return y_pred


class BinaryClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class MulticlassClassifier:
    strategies = ["one-vs-all", "all-vs-all"]

    def __init__(self, binary_classifier=BinaryClassifier, mode=None):
        self.mode = mode
        self.classifiers = [binary_classifier]
        self.classes = None
        self.subsamples = None

    @staticmethod
    def filter_data(key, data, target_name=None):
        if target_name is None:
            return data.iloc[:, -1].where(data.iloc[:, -1] == key, 1, 0, inplace=True)
        else:
            return data[target_name].where(data[target_name] == key, 1, 0, inplace=True)

    @staticmethod
    def take_subsample(classes, data, target_name=None):
        if target_name is None:
            data = data[data.iloc[:, -1] == classes[0] or data.iloc[:, -1] == classes[1]]
            return data.where(data.iloc[:, -1] == classes[0], 0, 1)
        else:
            data = data[data[target_name] == classes[0] or data[target_name] == classes[1]]
            return data.where(data[target_name] == classes[0], 0, 1)

    def fit(self, X, y):
        self.classes = np.unique(y)

        if self.mode == self.strategies[0]:
            self.classifiers *= len(self.classes)
            for i in range(len(self.classes)):
                data = MulticlassClassifier.filter_data(self.classes[i], X.copy())
                X, y = data.iloc[:, : -1], data.iloc[:, -1]
                self.classifiers[i].fit(X=X, y=y)

        elif self.mode == self.strategies[1]:
            num_of_classifiers = (len(self.classes) * (len(self.classes) + 1)) // 2
            self.classifiers *= num_of_classifiers
            cur_cls = 0
            for i in range(len(self.classes)):
                for j in range(i + 1, len(self.classes)):
                    data = MulticlassClassifier.take_subsample((i, j), X.copy())
                    X, y = data.iloc[:, : -1], data.iloc[:, -1]
                    self.classifiers[cur_cls].fit(X=X, y=y)
                    cur_cls += 1

    def _voting_of_classifiers(self, predictions, y_pred):
        for index, row in predictions.iterrows():
            classes = [0] * len(self.classes)
            lead_cls = 0
            for i in range(1, len(row)):
                classes[row[i]] += 1
                if classes[row[i]] > classes[lead_cls]:
                    lead_cls = row[i]
            y_pred[index] = lead_cls

    @staticmethod
    def _most_likely_class(y_proba, y_pred):
        for index, row in y_proba.iterrows():
            max_p = 0
            for cls in range(1, len(row)):
                if row[cls] > max_p:
                    max_p = row[cls]
                    y_pred[index] = cls

    def predict(self, X, threshold=0.5):
        y_pred = [None] * len(X)

        if self.mode == self.strategies[0]:
            y_proba = pd.DataFrame({'proba': [None] * len(X)})

            for cls in range(len(self.classifiers)):
                proba = pd.DataFrame({'class': self.classifiers[cls].predict(X=X)})
                y_proba = pd.concat([y_proba, proba], axis=1)

            self._most_likely_class(y_proba, y_pred)

        elif self.mode == self.strategies[1]:
            predictions = pd.DataFrame({'pred': [None] * len(X)})

            for k in range(len(self.classifiers)):
                proba = self.classifiers[k].predict(X=X)
                pred = pd.Series(np.where(proba < threshold, self.subsamples[k][0], self.subsamples[k][1]))
                predictions = pd.concat([predictions, pred], axis=1)

            self._voting_of_classifiers(predictions, y_pred)

        return y_pred
