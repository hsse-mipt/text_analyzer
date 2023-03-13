import pytest
from text_analyzer.src.metrics import binary_classification as metrics
import sklearn.metrics as skmetrics

import text_analyzer.src.models.baselines as baselines
from sklearn.dummy import DummyClassifier as skDummyClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from math import isclose
from random import randint


class TestMetrics:

    def test_binary_classification(self):
        y_true, y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0]), np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
        assert isclose(skmetrics.f1_score(y_true, y_pred), metrics.f1_score(y_true, y_pred))
        assert isclose(skmetrics.accuracy_score(y_true, y_pred), metrics.accuracy_score(y_true, y_pred))
        assert isclose(skmetrics.precision_score(y_true, y_pred), metrics.precision_score(y_true, y_pred))
        assert isclose(skmetrics.recall_score(y_true, y_pred), metrics.recall_score(y_true, y_pred))

        y_proba = np.array(list((randint(0, 1) / randint(1, 100) for _ in range(len(y_true)))))
        x, y, th = metrics.roc_curve(y_true, y_proba, 10000)
        sk_x, sk_y, sk_th = skmetrics.roc_curve(y_true, y_proba)
        assert isclose(metrics.auc(x, y), skmetrics.auc(sk_x, sk_y))
        assert isclose(metrics.roc_auc_score(y_true, y_proba), skmetrics.roc_auc_score(y_true, y_proba))


class TestModels:

    def test_dummy_classifier(self):
        X = pd.read_csv('../../static/datasets/modified/bin_classification/train_data.csv')
        y = pd.read_csv('../../static/datasets/modified/bin_classification/train_data.csv')
        y = y.label.to_list()
        classes = np.unique(X.label.to_list())

        w = X.label.value_counts().tolist()
        w = np.array(w) / sum(w)

        clf = baselines.DummyClassifier(classes, weights=w)
        y_pred = clf.predict(len(y))

        print(skmetrics.f1_score(y_true=y, y_pred=y_pred))
        print(skmetrics.accuracy_score(y_true=y, y_pred=y_pred))
        print(skmetrics.recall_score(y_true=y, y_pred=y_pred))

        clf = baselines.DummyClassifier(classes)
        y_pred = clf.predict(len(y))

        print(skmetrics.f1_score(y_true=y, y_pred=y_pred))
        print(skmetrics.accuracy_score(y_true=y, y_pred=y_pred))
        print(skmetrics.recall_score(y_true=y, y_pred=y_pred))

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

        # assert isclose(skmetrics.f1_score(test_labels, mnb.predict(test_feature_matrix), average='weighted'),
        #                skmetrics.f1_score(test_labels, nbclf.predict(test_feature_matrix), average='weighted'))

# %%
