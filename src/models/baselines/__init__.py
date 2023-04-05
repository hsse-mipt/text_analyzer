import numpy as np
import pandas as pd
import scipy
from math import log, exp, sqrt, pi
from random import choices
from abc import ABC, abstractmethod
from tqdm import tqdm


class DummyClassifier:
    def __init__(self, classes, weights=None):
        self.classes = classes
        self.weights = weights

    def predict(self, n_predictions=1):
        return choices(self.classes, weights=self.weights, k=n_predictions)


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
        
        def get_proba(self, value: float) -> float:
            if self.mode == 'gaussian':
                return self.get_prob_from_gauss(self, value, self.mean, self.deviation)
            if self.mode == 'gaussian_kde':
                return self.distribution.pdf(value)
            return self.distribution[value]
        
        @staticmethod
        def get_prob_from_gauss(self, value: float, mean: float, deviation: float) -> float:
            if deviation == 0:
                return 1
            return exp(-((value - mean) ** 2) / (2 * deviation ** 2)) / sqrt(2 * pi * deviation ** 2)
    
    def __init__(self, mode = 'gaussian_kde'): #guassian_kde, guassian, cat_feautures
        self.mode = mode
    
    
    def fit(self, train_feature_matrix, train_labels):
        self.unique_labels = np.unique(train_labels)
        self.class_probability = {} # здесь будем хранить долю каждого класса в выборке
        self.distributions = {} # key: label, value: []
                                # [i] - плотность распределения i-й фичи в X при условии, что класс = label, т.е. плотность распределения P(x_i | label))
        for label in self.unique_labels:
            current_label_feature_matrix = train_feature_matrix[np.array(train_labels == label)] # выбираем те строки, для которых класс = label
            self.class_probability[label] = current_label_feature_matrix.size / train_feature_matrix.size
            self.distributions[label] = []
            for feature_ind in range(train_feature_matrix.shape[1]):
                feature_column = current_label_feature_matrix.iloc[:, feature_ind] # выбираем фичу с номером feature_ind
                distribution = self.Distribution(feature_column, self.mode)  # считаем плотность распределения P(x_feature_ind | label) с помощью gaussian_kde
                self.distributions[label].append(distribution)
    
    def predict(self, test_feature_matrix):
        y_pred = [] # наши предсказания
        for i, features in test_feature_matrix.iterrows():
            max_likelihood = -float('inf')
            for label in self.unique_labels: # перебираем возможные варианты ответа, выбираем - максимально правдоподобный
                likelihood = log(self.class_probability[label]) # здесь - сумма логарифмов вероятностей для label
                for feature_ind in range(test_feature_matrix.shape[1]):
                    likelihood += log(self.distributions[label][feature_ind].get_proba(float(features[feature_ind])))
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    predict = label
            y_pred.append(predict)
        return y_pred

def filter_data(key, data, target_name=None):
    if target_name is None:
        return data.iloc[-1].where(data.iloc[-1] == key, 1, 0, inplace=True)
    else:
        return data[target_name].where(data[target_name] == key, 1, 0, inplace=True)


def take_subsample(classes, data, target_name=None):
    if target_name is None:
        data = data[data.iloc[-1] == classes[0] or data.iloc[-1] == classes[1]]
        return data.where(data.iloc[-1] == classes[0], 0, 1)
    else:
        data = data[data[target_name] == classes[0] or data[target_name] == classes[1]]
        return data.where(data[target_name] == classes[0], 0, 1)


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
        self.bin_clf = binary_classifier
        self.mode = mode
        self.classifiers = [self.bin_clf]
        self.classes = None
        self.subsamples = None

    def fit(self, X, y):
        self.classes = np.unique(y)

        if self.mode == self.strategies[0]:
            self.classifiers *= len(self.classes)
            for i in range(len(self.classes)):
                data = filter_data(self.classes[i], X.copy())
                X, y = data[:-1], data[-1]
                self.classifiers[i].fit(X=X, y=y)

        elif self.mode == self.strategies[1]:
            num_of_classifiers = (len(self.classes) * (len(self.classes) + 1)) // 2
            self.classifiers *= num_of_classifiers
            self.subsamples = []
            cur_cls = 0
            for i in range(len(self.classes)):
                for j in range(i + 1, len(self.classes)):
                    self.subsamples.append((i, j))
                    data = take_subsample((i, j), X.copy())
                    X, y = data[:-1], data[-1]
                    self.classifiers[cur_cls].fit(X=X, y=y)
                    cur_cls += 1

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

#%%
