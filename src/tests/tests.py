import pytest
from text_analyzer.src.metrics import binary_classification as bin_metrics
from text_analyzer.src.metrics import multiclass_classification as multi_metrics
import sklearn.metrics as skmetrics

import text_analyzer.src.models.baselines as baselines
from sklearn.dummy import DummyClassifier as skDummyClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
from math import isclose
from random import randint


class TestMetrics:

    def test_binary_classification(self):
        y_true, y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0]), np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
        assert isclose(skmetrics.f1_score(y_true, y_pred), bin_metrics.f1_score(y_true, y_pred))
        assert isclose(skmetrics.accuracy_score(y_true, y_pred), bin_metrics.accuracy_score(y_true, y_pred))
        assert isclose(skmetrics.precision_score(y_true, y_pred), bin_metrics.precision_score(y_true, y_pred))
        assert isclose(skmetrics.recall_score(y_true, y_pred), bin_metrics.recall_score(y_true, y_pred))

        y_proba = np.array(list((randint(0, 1) / randint(1, 100) for _ in range(len(y_true)))))
        x, y, th = bin_metrics.roc_curve(y_true, y_proba, 10000)
        sk_x, sk_y, sk_th = skmetrics.roc_curve(y_true, y_proba)
        assert isclose(bin_metrics.auc(x, y), skmetrics.auc(sk_x, sk_y))
        assert isclose(bin_metrics.roc_auc_score(y_true, y_proba), skmetrics.roc_auc_score(y_true, y_proba))

    def test_multiclass_classification(self):
        y_true, y_pred = np.array([0, 1, 1, 2, 0, 1, 1, 0, 0, 2]), np.array([0, 1, 1, 2, 0, 1, 0, 1, 2, 2])
        assert isclose(skmetrics.f1_score(y_true, y_pred, average="micro"),
                       multi_metrics.f1_score(y_true, y_pred, average="micro"))
        # assert isclose(skmetrics.accuracy_score(y_true, y_pred),
        #                multi_metrics.accuracy_score(y_true, y_pred, average="micro"))
        assert isclose(skmetrics.precision_score(y_true, y_pred, average="micro"),
                       multi_metrics.precision_score(y_true, y_pred, average="micro"))
        assert isclose(skmetrics.recall_score(y_true, y_pred, average="micro"),
                       multi_metrics.recall_score(y_true, y_pred, average="micro"))

        assert isclose(skmetrics.f1_score(y_true, y_pred, average="macro"),
                       multi_metrics.f1_score(y_true, y_pred, average="macro"))
        # assert isclose(skmetrics.accuracy_score(y_true, y_pred),
        #                multi_metrics.accuracy_score(y_true, y_pred, average="macro"))
        assert isclose(skmetrics.precision_score(y_true, y_pred, average="macro"),
                       multi_metrics.precision_score(y_true, y_pred, average="macro"))
        assert isclose(skmetrics.recall_score(y_true, y_pred, average="macro"),
                       multi_metrics.recall_score(y_true, y_pred, average="macro"))


# class TestModels:
#
#     def test_dummy_classifier(self):
#         X, y = datasets.load_iris(return_X_y=True)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#         classes = np.unique(y_train)
#         weights = [0.33, 0.33, 0.33]
#
#         skclf = skDummyClassifier(strategy='uniform')
#         skclf.fit(X_train, y_train)
#         clf = baselines.DummyClassifier(classes, weights)
#         skclf.fit(X_train, y_train)
#
#         # assert (isclose(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
#         #                 skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted")))
#
#         print(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
#               skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted"))
#
#         skclf = skDummyClassifier(strategy='stratified')
#         skclf.fit(X_train, y_train)
#         clf = baselines.DummyClassifier(classes)
#
#         # assert (isclose(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
#         #                 skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted")))
#
#         print(skmetrics.f1_score(y_test, clf.predict(n_predictions=len(X_test)), average="weighted"),
#               skmetrics.f1_score(y_test, skclf.predict(X_test), average="weighted"))
#
#     def test_naive_bayes_classifier(self):
#         dataset = datasets.load_iris()
#         feature_matrix = dataset.data
#         labels = dataset.target
#
#         train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
#             feature_matrix, labels, test_size=0.2, random_state=42)
#
#         mnb = MultinomialNB()
#         mnb.fit(train_feature_matrix, train_labels)
#
#         nbclf = baselines.NaiveBayesClassifier()
#         nbclf.fit(train_feature_matrix, train_labels)
#
#         # assert isclose(skmetrics.f1_score(test_labels, mnb.predict(test_feature_matrix), average='weighted'),
#         #                skmetrics.f1_score(test_labels, nbclf.predict(test_feature_matrix), average='weighted'))
