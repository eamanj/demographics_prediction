#!/usr/bin/python

import sys
import argparse
import utils

from sklearn import svm, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score


# function returns a custom callable scorer object to be used by grid search scoring. name
# identifies the type of the scorer to create. Possible names are accuracy,
# weighted-precision, macro-precision, weighted-recall, macro-recall, weighted-f1,
# macro-f1. Theses scorers are different from default version in that they are initialized
# with additional and non-default parameters.
def create_scorer(name):
  if name == "accuracy":
    return make_scorer(accuracy_score)
  elif name == "weighted-precision":
    return make_scorer(precision_score, pos_label = None, average = 'weighted')
  elif name == "macro-precision":
    return make_scorer(precision_score, pos_label = None, average = 'macro')
  elif name == "weighted-recall":
    return make_scorer(recall_score, pos_label = None, average = 'weighted')
  elif name == "macro-recall":
    return make_scorer(recall_score, pos_label = None, average = 'macro')
  elif name == "weighted-f1":
    return make_scorer(f1_score, pos_label = None, average = 'weighted')
  elif name == "macro-f1":
    return make_scorer(f1_score, pos_label = None, average = 'macro')
  else:
    sys.exit('Invalid scoring function: ' + scoring_function + ' provided')


def grid_search(score,
                features,
                labels,
                scikit_balancing,
                algorithm,
                num_jobs):
  """
  expects the features to be scaled!
  """
  # Now balance the train data set and create requested train size.
  features, labels, penalty_weights = utils.prepare_train_data(
      features, labels, scikit_balancing, -1)

  # Set the parameters for gid search and model based on algorithm choice
  if algorithm == 'kernel-svm':
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 0.0001],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [0.1, 0.01, 0.001, 0.0001],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel' : ['poly'], 'degree': [2, 3, 4],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    model = svm.SVC(tol=0.005, cache_size=6000, class_weight=penalty_weights)
  elif algorithm == 'linear-svm':
    tuned_parameters = [{'loss': ['hinge', 'squared_hinge'],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    model = svm.LinearSVC(tol=0.005, max_iter=5000, class_weight=penalty_weights)
  elif algorithm == 'logistic':
    tuned_parameters = [{
      'penalty': ['l1', 'l2'],
      'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}]
    model = LogisticRegression(tol=0.0005, max_iter=1000, class_weight=penalty_weights)
  elif algorithm == 'random-forest':
    tuned_parameters = [{'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 'log2', 0.5, 0.8],
                         'min_samples_split': [2], 'min_samples_leaf': [1]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 'log2', 0.5, 0.8],
                         'min_samples_split': [5], 'min_samples_leaf': [1, 2]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 'log2', 0.5, 0.8],
                         'min_samples_split': [10], 'min_samples_leaf': [2, 5]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 'log2', 0.5, 0.8],
                         'min_samples_split': [20], 'min_samples_leaf': [5, 10]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 'log2', 0.5, 0.8],
                         'min_samples_split': [50], 'min_samples_leaf': [5, 15, 25]}]
    model = RandomForestClassifier(class_weight=penalty_weights)
  elif algorithm == 'knn':
    tuned_parameters = [{'n_neighbors': [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 70, 100,
                                         150, 200],
                         'metric': ['euclidean', 'manhattan', 'chebyshev'],
                         'algorithm': ['ball_tree', 'kd_tree'],
                         'weights': ['uniform', 'distance']}]
    model = KNeighborsClassifier()
  else:
    sys.exit('Invalid algorithm: ' + algorithm + ' provided')

  scorer = create_scorer(score)
  skf = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle = True)
  # Don't pre dispatch all jobs at once, only dispatch ones you are runnings so memory
  # usage does not blow up
  clf = GridSearchCV(estimator=model,
                     param_grid=tuned_parameters,
                     n_jobs=num_jobs,
                     pre_dispatch="n_jobs",
                     cv=skf,
                     scoring=scorer)

  clf.fit(features, labels)
  return clf
