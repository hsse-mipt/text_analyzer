import pytest
from text_analyzer.src.metrics import binary_classification as metrics
import sklearn.metrics as skmetrics

import text_analyzer.src.models.baselines as baselines
from sklearn.dummy import DummyClassifier as skDummyClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
from math import isclose
import random


class TestMetrics:

    def test_binary_classification(self):
        y_true, y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0]), np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
        assert isclose(skmetrics.f1_score(y_true, y_pred),
                       metrics.f1_score(y_true, y_pred))
        assert isclose(skmetrics.accuracy_score(y_true, y_pred),
                       metrics.accuracy_score(y_true, y_pred))
        assert isclose(skmetrics.precision_score(y_true, y_pred),
                       metrics.precision_score(y_true, y_pred))
        assert isclose(skmetrics.recall_score(y_true, y_pred),
                       metrics.recall_score(y_true, y_pred))

        y_proba = np.array(list((random.randint(0, 1) for _ in range(len(y_true)))))
        x, y, th = metrics.roc_curve(y_true, y_proba)
        sk_x, sk_y, sk_th = skmetrics.roc_curve(y_true, y_proba)
        # assert isclose(skmetrics.auc(x, y), skmetrics.auc(sk_x, sk_y))  #тест roc_curve
        # assert isclose(metrics.auc(x, y), skmetrics.auc(sk_x, sk_y))    #тест auc


class TestModels:

    def test_dummy_classifier(self):
        X, y = datasets.load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        classes = np.unique(y_train)
        weights = [0.33, 0.33, 0.33]

        skclf = skDummyClassifier(strategy='uniform')
        skclf.fit(X_train, y_train)
        clf = baselines.DummyClassifier(classes, weights)
        skclf.fit(X_train, y_train)

        # assert (isclose(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
        #                 skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted")))

        print(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
              skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted"))

        skclf = skDummyClassifier(strategy='stratified')
        skclf.fit(X_train, y_train)
        clf = baselines.DummyClassifier(classes)

        # assert (isclose(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
        #                 skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted")))

        print(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
              skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted"))

    def test_naive_bayes_classifier(self):
        dataset = datasets.load_iris()
        feature_matrix = dataset.data
        labels = dataset.target

        train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
            feature_matrix, labels, test_size=0.2, random_state=42)

        mnb = MultinomialNB()
        mnb.fit(train_feature_matrix, train_labels)

        nbclf = baselines.NaiveBayesClassifier()
        nbclf.fit(train_feature_matrix, train_labels)

        assert isclose(skmetrics.f1_score(test_labels, mnb.predict(test_feature_matrix), average='weighted'),
                       skmetrics.f1_score(test_labels, nbclf.predict(test_feature_matrix), average='weighted'))
