import numpy as np
import pandas as pd
import scipy
from math import log, exp, sqrt, pi
from random import choices
from abc import ABC, abstractmethod
from tqdm import tqdm


class DummyClassifier:
    def __init__(self):
        self.classes = None
        self.weights = None

    def get_weights(self, y) -> list:
        occurrences = {}
        for y_i in self.classes:
            occurrences[y_i] = y.count(y_i) / len(y)
        return list(occurrences.values())

    def fit(self, X, y, weights=None):
        self.classes = np.unique(y)
        self.weights = self.get_weights(list(y)) if weights is None else weights

    def predict(self, X):
        return choices(self.classes, weights=self.weights, k=X.shape[0])


class NaiveBayesClassifier:
    class Distribution:
        def __init__(self, feature_column, mode: str):
            self.mode = mode
            if mode == 'gaussian_kde':
                self.distribution = scipy.stats.gaussian_kde(feature_column)
            if mode == 'gaussian':
                self.mean = feature_column.mean()
                self.deviation = np.std(feature_column)
            if mode == 'cat_features':
                unique, counts = np.unique(feature_column, return_counts=True)
                self.distribution = dict(zip(unique, counts / feature_column.size))
                self.distribution = dict(zip(unique, counts / feature_column.size))

        def get_proba(self, value: float) -> float:
            if self.mode == 'gaussian':
                return self.get_prob_from_gauss(self, value, self.mean, self.deviation)
            if self.mode == 'gaussian_kde':
                return self.distribution.pdf(value)
            return self.distribution[value]

        @staticmethod
        def get_prob_from_gauss(self, value: float, mean: float, deviation: float) -> float:
            '''
            Вычисляет сложную формулу для вероятности из нормального распределения
            '''
            if deviation == 0:
                return 1
            return exp(-((value - mean) ** 2) / (2 * deviation ** 2)) / sqrt(
                2 * pi * deviation ** 2)

        def __str__(self):
            return str(self.distribution)

    def __init__(self, mode='gaussian_kde'):  # guassian_kde, guassian, cat_feautures
        self.distributions = None
        self.class_probability = None
        self.unique_labels = None
        self.mode = mode

    def fit(self, train_feature_matrix, train_labels):
        self.unique_labels = np.unique(train_labels)
        self.class_probability = {}  # здесь будем хранить долю каждого класса в выборке
        self.distributions = {}  # key: label, value: []
        # [i] - плотность распределения i-й фичи в X при условии, что класс = label, т.е. плотность распределения P(x_i | label))
        for label in self.unique_labels:
            current_label_feature_matrix = train_feature_matrix[
                np.array(train_labels == label)]  # выбираем те строки, для которых класс = label
            self.class_probability[
                label] = current_label_feature_matrix.size / train_feature_matrix.size
            self.distributions[label] = []
            for feature_ind in range(train_feature_matrix.shape[1]):
                feature_column = current_label_feature_matrix.iloc[:,
                                 feature_ind]  # выбираем фичу с номером feature_ind
                distribution = self.Distribution(feature_column,
                                                 self.mode)  # считаем плотность распределения P(x_feature_ind | label) с помощью gaussian_kde
                self.distributions[label].append(distribution)

    def predict(self, test_feature_matrix):
        y_pred = []  # наши предсказания
        for i, features in test_feature_matrix.iterrows():
            max_likelihood = -float('inf')
            max_probabilities = []
            for label in self.unique_labels:  # перебираем возможные варианты ответа, выбираем - максимально правдоподобный
                likelihood = log(self.class_probability[
                                     label])  # здесь - сумма логарифмов вероятностей для label
                probabilities = [self.class_probability[label]]
                for feature_ind in range(test_feature_matrix.shape[1]):
                    likelihood += log(self.distributions[label][feature_ind].get_proba(
                        float(features[feature_ind])))
                    if self.mode == 'cat_features' and features[feature_ind] not in \
                            self.distributions[label][feature_ind].distribution.keys() or \
                            self.distributions[label][feature_ind].get_proba(
                                features[feature_ind]) == 0:
                        likilihood = -float('inf')
                        probabilities.append(-float('inf'))
                    else:
                        likelihood += log(
                            self.distributions[label][feature_ind].get_proba(features[feature_ind]))
                        probabilities.append(
                            self.distributions[label][feature_ind].get_proba(features[feature_ind]))
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    max_probabilities = probabilities
                    predict = label
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
    def filter_data(key, data, target_name=None):
        if target_name is None:
            target = data.iloc[:, -1:].apply(lambda x: x == key).astype(int)
            return pd.concat([data.iloc[:, :-1], target], axis=1)
        else:
            target = data[target_name].apply(lambda x: x == key).astype(int)
            return pd.concat([data.drop(columns=target_name), target], axis=1)

    @staticmethod
    def take_subsample(classes, data, target_name=None):
        if target_name is None:
            # data = data[data[:, -1].isin(classes)]  TODO this version doesn't work
            target = data.iloc[:, -1:].apply(lambda x: x == classes[1]).astype(int)
            return pd.concat([data.iloc[:, :-1], target], axis=1)
        else:
            data = data[data[target_name].isin(classes)]
            target = data[target_name].apply(lambda x: x == classes[1]).astype(int)
            return pd.concat([data.drop(columns=target_name), target], axis=1)

    def make_array_of_classifiers(self, n):
        for clf_ in range(n):
            self.classifiers.append(self.clf())

    def fit(self, X, y, target_name=None):
        self.classes = np.unique(y).astype(int)
        train_data = pd.concat([X, y], axis=1)

        if self.mode == self.strategies[0]:
            self.make_array_of_classifiers(len(self.classes))

            for i in range(len(self.classes)):
                data = MulticlassClassifier.filter_data(self.classes[i], train_data, target_name)
                X_train, y_train = data.iloc[:, :-1], np.ravel(data.iloc[:, -1:])
                self.classifiers[i].fit(X_train, y_train)

        elif self.mode == self.strategies[1]:
            num_of_classifiers = (len(self.classes) * (len(self.classes) - 1)) // 2
            self.make_array_of_classifiers(num_of_classifiers)

            cur_cls = 0
            self.sub_samples = [[None] * 2 for _ in range(num_of_classifiers)]
            for i in range(len(self.classes)):
                for j in range(i + 1, len(self.classes)):
                    self.sub_samples[cur_cls][0], self.sub_samples[cur_cls][1] = self.classes[i], \
                    self.classes[j]
                    data = MulticlassClassifier.take_subsample((self.classes[i], self.classes[j]),
                                                               train_data, target_name)
                    X_train, y_train = data.iloc[:, :-1], np.ravel(data.iloc[:, -1:])
                    self.classifiers[cur_cls].fit(X_train, y_train)
                    cur_cls += 1

    def _most_likely_class(self, y_proba, y_pred):
        for i, proba in y_proba.iterrows():
            max_proba = 0
            for y_i in range(len(proba)):
                if proba[y_i] >= max_proba:
                    max_proba = proba[y_i]
                    y_pred[i] = self.classes[y_i]

    def predict(self, X):
        y_pred = [None for _ in range(len(X))]

        if self.mode == self.strategies[0]:
            y_proba = pd.DataFrame({'proba': [None for _ in range(len(X))]})

            for cls in range(len(self.classifiers)):
                proba = pd.DataFrame({'class': self.classifiers[cls].predict(X)})
                y_proba = pd.concat([y_proba, proba], axis=1)

            y_proba.drop(columns=['proba'], inplace=True)

            self._most_likely_class(y_proba, y_pred)

        elif self.mode == self.strategies[1]:
            predictions = pd.DataFrame({'pred': [None for _ in range(len(X))]})

            for k in range(len(self.classifiers)):
                proba = pd.DataFrame({'proba': self.classifiers[k].predict(X)})
                proba = proba.applymap(lambda p: self.sub_samples[k][p])
                proba.rename(columns={'proba': f'pred_{k}'}, inplace=True)
                predictions = pd.concat([predictions, proba[f'pred_{k}']], axis=1)

            predictions.drop(columns=['pred'], inplace=True)

            self._voting_of_classifiers(predictions, y_pred)

        return y_pred

    def _voting_of_classifiers(self, predictions, y_pred):
        for i, y in predictions.iterrows():
            classes = dict((y_i, 0) for y_i in self.classes)
            lead_y = 0
            for y_i in y:
                classes[y_i] += 1
                if classes[y_i] > classes[lead_y]:
                    lead_y = y_i
            y_pred[i] = lead_y
