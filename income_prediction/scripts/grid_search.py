#!/usr/bin/python

import sys
import argparse
import utils
import numpy

from mord import LogisticIT, LogisticAT, LogisticSE, LAD, OrdinalRidge
from sklearn import svm, model_selection
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def grid_search(evaluation,
                features,
                labels,
                penalty_weights,
                algorithm,
                num_jobs,
                **options):
  """
  expects the features to be scaled for svm and knn.
  """
  # Set the parameters for gid search and model based on algorithm choice
  if algorithm == 'kernel-svm':
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 0.0001],
                         'decision_function_shape': ['ovo', 'ovr'],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [0.1, 0.01, 0.001, 0.0001],
                         'decision_function_shape': ['ovo', 'ovr'],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                        {'kernel' : ['poly'], 'degree': [2, 3],
                         'decision_function_shape': ['ovo', 'ovr'],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    model = svm.SVC(tol=0.05, cache_size=6000, class_weight=penalty_weights)

  elif algorithm == 'linear-svm':
    tuned_parameters = [{'loss': ['hinge', 'squared_hinge'],
                         'multi_class': ['ovr'],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    model = svm.LinearSVC(tol=0.05, max_iter=5000, class_weight=penalty_weights)

  elif algorithm == 'logistic':
    # newton, lbfgs only support L2
    costs_list = (10.0**numpy.arange(-6,5)).tolist()
    tuned_parameters = [
      {'multi_class': ['ovr'],
       'solver': ['liblinear'],
       'penalty': ['l1', 'l2'],
       'C': costs_list},
      {'multi_class': ['multinomial'],
       'solver': ['lbfgs'],
       'penalty': ['l2'],
       'C': costs_list}]
    model = LogisticRegression(tol=0.005, max_iter=5000, class_weight=penalty_weights)

  elif algorithm == 'sgd-logistic':
    alphas_list = (10.0**numpy.arange(-8,1)).tolist()
    tuned_parameters = [{'penalty': ['l1', 'l2'],
                         'alpha': alphas_list},
                        {'penalty': ['elasticnet'],
                         'alpha': alphas_list,
                         'l1_ratio': [0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]}]
    # loss should be log for logistic classifier. We don't set n_jobs since grid search
    # will use the cores
    n_iter = numpy.ceil(5*(10**6) / features.shape[0])
    model = SGDClassifier(loss='log', class_weight=penalty_weights,
                          n_iter=n_iter, n_jobs=1)

  elif algorithm == 'sgd-svm':
    alphas_list = (10.0**numpy.arange(-8,1)).tolist()
    tuned_parameters = [{'penalty': ['l1', 'l2'],
                         'alpha': alphas_list},
                        {'penalty': ['elasticnet'],
                         'alpha': alphas_list,
                         'l1_ratio': [0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]}]
    # loss should be hinge for linear svm classifier. We don't set n_jobs since grid
    # search will use the cores
    n_iter = numpy.ceil(5*(10**6) / features.shape[0])
    model = SGDClassifier(loss='hinge', class_weight=penalty_weights,
                          n_iter=n_iter, n_jobs=1)

  elif algorithm == 'random-forest':
    tuned_parameters = [{'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 0.4, 0.8],
                         'min_samples_split': [2], 'min_samples_leaf': [1]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 0.4, 0.8],
                         'min_samples_split': [5], 'min_samples_leaf': [1, 2]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 0.4, 0.8],
                         'min_samples_split': [10], 'min_samples_leaf': [2, 5]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 0.4, 0.8],
                         'min_samples_split': [20], 'min_samples_leaf': [5, 10]},
                        {'n_estimators': [100], 'criterion': ['gini', 'entropy'],
                         'max_features': ['sqrt', 0.4, 0.8],
                         'min_samples_split': [50], 'min_samples_leaf': [5, 15, 25]}]
    model = RandomForestClassifier(class_weight=penalty_weights)

  elif algorithm == 'knn':
    tuned_parameters = [{'n_neighbors': [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 70, 100,
                                         150, 200],
                         'metric': ['euclidean', 'manhattan', 'chebyshev'],
                         'algorithm': ['ball_tree', 'kd_tree'],
                         'weights': ['uniform', 'distance']}]
    model = KNeighborsClassifier()

  elif algorithm == 'ridgeclassifier':
    alphas_list = (10.0**numpy.arange(-5,5)).tolist()
    tuned_parameters = [{'alpha': alphas_list,
                         'normalize': [True, False]}]
    model = RidgeClassifier(max_iter=10000, class_weight=penalty_weights)

  elif algorithm == 'logisticse':
    alphas_list = (10.0**numpy.arange(-5,5)).tolist()
    tuned_parameters = [{'alpha': alphas_list}]
    model = LogisticSE(max_iter=10000)

  elif algorithm == 'logisticit':
    alphas_list = (10.0**numpy.arange(-5,5)).tolist()
    tuned_parameters = [{'alpha': alphas_list}]
    model = LogisticIT(max_iter=10000)

  elif algorithm == 'logisticat':
    alphas_list = (10.0**numpy.arange(-5,5)).tolist()
    tuned_parameters = [{'alpha': alphas_list}]
    model = LogisticAT(max_iter=10000)

  elif algorithm == 'ordinalridge':
    alphas_list = (10.0**numpy.arange(-5,5)).tolist()
    tuned_parameters = [{'alpha': alphas_list}]
    model = OrdinalRidge(max_iter=10000)

  elif algorithm == 'lad':
    tuned_parameters = [{'loss': ['l1', 'l2'],
                         'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
    model= LAD(max_iter=3000)

  else:
    sys.exit('Invalid algorithm: ' + algorithm + ' provided')

  scorer = utils.create_scorer(evaluation)
  skf = model_selection.StratifiedKFold(n_splits=5, shuffle = True)
  # Don't pre dispatch all jobs at once, only dispatch ones you are runnings so memory
  # usage does not blow up
  clf = GridSearchCV(estimator=model,
                     param_grid=tuned_parameters,
                     n_jobs=num_jobs,
                     pre_dispatch="n_jobs",
                     cv=skf,
                     scoring=scorer)

  clf.fit(features, labels)
  print "Best Grid Search Parameters are: " + str(clf.best_params_)
  print "Best Grid Search CV Score: " + str(clf.best_score_)
  
  return clf
