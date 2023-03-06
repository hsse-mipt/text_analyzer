import numpy as np
import pandas as pd
import scipy
from math import log
from random import choices


class DummyClassifier:
    def __init__(self, classes, weights=None):
        self.classes = classes
        self.weights = weights

    def predict(self, n_predictions=1):
        return choices(self.classes, weights=self.weights, k=n_predictions)


class NaiveBayesClassifier:
    def __init__(self):
        self.distributions = {}         # key: label, value: []
                                        # [i] - плотность распределения i-й фичи в X при условии, что класс = label, т.е. плотность распределения P(x_i | label))
        self.class_probability = {}     # здесь будем хранить долю каждого класса в выборке
        self.unique_labels = None

    def fit(self, train_feature_matrix, train_labels):
        self.unique_labels = np.unique(train_labels)
        for label in self.unique_labels:
            current_label_feature_matrix = train_feature_matrix[train_labels == label]      # выбираем те строки, для которых класс = label
            self.class_probability[label] = current_label_feature_matrix.size / train_feature_matrix.size
            self.distributions[label] = []
            for feature_ind in range(train_feature_matrix.shape[1]):
                feature_column = current_label_feature_matrix[feature_ind]      # выбираем фичу с номером feature_ind
                distribution = scipy.stats.gaussian_kde(feature_column)         # считаем плотность распределения P(x_feature_ind | label) с помощью gaussian_kde
                self.distributions[label].append(distribution)

    def predict(self, test_feature_matrix):
        y_pred = []                                             # наши предсказания
        predict = None
        for features in test_feature_matrix:
            max_likelihood = -float('inf')
            for label in self.unique_labels:                    # перебираем возможные варианты ответа, выбираем - максимально правдоподобный
                likelihood = log(self.class_probability[label]) # сумма логарифмов вероятностей для label
                for feature_ind in range(test_feature_matrix.shape[1]):
                    likelihood += self.distributions[label][feature_ind].logpdf(features[feature_ind])
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    predict = label
            y_pred.append(predict)
        return y_pred
