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

    def fit(self, X, y, weights=None):
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
        self.clf = binary_classifier
        self.classes = []
        self.sub_samples = []
        self.classifiers = []

    @staticmethod
    def filter_data(key, data):
        target = data.iloc[:, -1:].apply(lambda x: x == key).astype(int)
        return pd.concat([data.iloc[:, :-1], target], axis=1)

    @staticmethod
    def take_subsample(classes, data):
        data = data[data['label'].isin(classes)]    #TODO make via indexes, not col_name
        target = data.iloc[:, -1:].apply(lambda x: x == classes[1]).astype(int)
        return pd.concat([data.iloc[:, :-1], target], axis=1)

    def make_array_of_classifiers(self, n):
        for clf_ in range(n):
            self.classifiers.append(self.clf())

    def fit(self, X, y):
        self.classes = np.unique(y)
        train_data = pd.concat([X, y], axis=1)

        if self.mode == self.strategies[0]:
            self.make_array_of_classifiers(len(self.classes))

            for i in range(len(self.classes)):
                data = MulticlassClassifier.filter_data(self.classes[i], train_data)
                X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1:]
                self.classifiers[i].fit(X_train, y_train)

        elif self.mode == self.strategies[1]:
            num_of_classifiers = (len(self.classes) * (len(self.classes) + 1)) // 2
            self.make_array_of_classifiers(num_of_classifiers)

            cur_cls = 0
            self.sub_samples = [[None] * 2] * num_of_classifiers
            for i in range(len(self.classes)):
                for j in range(i + 1, len(self.classes)):
                    self.sub_samples[cur_cls][0], self.sub_samples[cur_cls][1] = i, j
                    data = MulticlassClassifier.take_subsample((i, j), train_data)
                    X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1:]
                    self.classifiers[cur_cls].fit(X_train, y_train)
                    cur_cls += 1

    @staticmethod
    def _most_likely_class(y_proba, y_pred):
        for i, proba in y_proba.iterrows():
            max_proba = 0
            for y_i in range(len(proba)):
                if proba[y_i] >= max_proba:
                    max_proba = proba[y_i]
                    y_pred[i] = y_i

    def predict(self, X, threshold=0.5):
        y_pred = [None] * len(X)

        if self.mode == self.strategies[0]:
            y_proba = pd.DataFrame({'proba': [None] * len(X)})

            for cls in range(len(self.classifiers)):
                proba = pd.DataFrame({'class': self.classifiers[cls].predict(X)})
                y_proba = pd.concat([y_proba, proba], axis=1)

            y_proba.drop(columns=['proba'], inplace=True)

            MulticlassClassifier._most_likely_class(y_proba, y_pred)

        elif self.mode == self.strategies[1]:
            predictions = pd.DataFrame({'pred': [None] * len(X)})

            for k in range(len(self.classifiers)):
                proba = pd.DataFrame(self.classifiers[k].predict(X))
                #pred = proba.apply(lambda x: self.sub_samples[k][0] if x < threshold else self.sub_samples[k][1]) TODO
                #predictions = pd.concat([predictions, pd.DataFrame(pred)], axis=1)

            predictions.drop(columns=['pred'], inplace=True)

            self._voting_of_classifiers(predictions, y_pred)

        return y_pred

    def _voting_of_classifiers(self, predictions, y_pred):
        for i, y in predictions.iterrows():
            classes = [0] * len(self.classes)
            lead_y = 0
            for y_i in y:
                classes[y_i] += 1
                if classes[y_i] > classes[lead_y]:
                    lead_y = y_i
            y_pred[i] = lead_y
