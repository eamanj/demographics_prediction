#!/usr/bin/python

# this file includes the methods that train svm, rf, logistic and knn models and return
# them

import numpy
import utils
import grid_search
import feature_selection
import mord
from sklearn import svm
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_logistic(train_features,
                   train_labels,
                   penalty_weights,
                   skip_grid_search,
                   evaluation,
                   num_jobs,
                   penalty,
                   cost,
                   multi_class,
                   solver):
  """
  returns the trained logistic model.  penalty, cost, multi_class and solver are ignored
  if grid search is requested.
  """
  # requested grid search. find best parameters, to achieve highest average score
  if not skip_grid_search:
    algorithm = "logistic"
    clf = grid_search.grid_search(evaluation,
                                  train_features,
                                  train_labels,
                                  penalty_weights,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    penalty = params['penalty']
    cost = params['C']
    multi_class = params['multi_class']
    solver = params['solver']

  # Now perform the training on full train data.
  model = LogisticRegression(penalty=penalty,
                             C=cost,
                             multi_class=multi_class,
                             solver=solver,
                             max_iter=20000,
                             class_weight=penalty_weights)
  model = model.fit(train_features, train_labels)
  
  return model


def train_sgd(train_features,
              train_labels,
              penalty_weights,
              skip_grid_search,
              evaluation,
              algorithm,
              num_jobs,
              penalty,
              alpha,
              l1_ratio):
  """
  returns the trained sgd model. The model will be SVM or logistic based on the value of
  algorithm.
  algorithm: should be either sgd-svm or sgd-logistic.  The loss in the SGD Classifier
  must be log for logistic regreesion and hinge for SVM. penalty, alpha and l1_ratio are
  ignored if grid search is requested.
  
  l1_ratio is the Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0
  corresponds to L2 penalty, l1_ratio=1 to L1. It determines the level of feature
  selection.

  penalty determines how to perform grid search.  choices l1,l2,elasticnet. l2 might take
  a long time in with large feature set.
  """
  if algorithm == 'sgd-logistic':
    loss = 'log'
  elif algorithm == 'sgd-svm':
    loss = 'hinge'
  else:
    sys.exit('bad algorithm provided to sgd model.')
  
  # requested grid search. find best parameters, to achieve highest average score
  if not skip_grid_search:
    clf = grid_search.grid_search(evaluation,
                                  train_features,
                                  train_labels,
                                  penalty_weights,
                                  algorithm,
                                  num_jobs)
                                 
    params = clf.best_params_
    penalty = params['penalty']
    alpha = params['alpha']
    if 'l1_ratio' in params:
      l1_ratio = params['l1_ratio']
  # Now perform the training on full train data.
  n_iter = numpy.ceil(5*(10**6) / train_features.shape[0])
  model = SGDClassifier(loss=loss,
                        penalty=penalty,
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        class_weight=penalty_weights,
                        n_iter=n_iter,
                        n_jobs=num_jobs)
  model = model.fit(train_features, train_labels)
  
  return model


def train_knn(train_features,
              train_labels,
              skip_grid_search,
              evaluation,
              num_jobs,
              n_neighbors,
              weights,
              algorithm,
              metric):
  """
  returns the trained knn model. num_neighbors, weights, algorithm and metric are ignored
  if grid search is requested.
  Assumes the features are scaled!
  """
  # knn does not take in penalty based on weight
  penalty_weights = 'knn-does-not-require-weights-on-imbalanced-data'
  if not skip_grid_search:
    algorithm = "knn"
    clf = grid_search.grid_search(evaluation,
                                  train_features,
                                  train_labels,
                                  penalty_weights,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    n_neighbors = params['n_neighbors']
    weights = params['weights']
    algorithm = params['algorithm']
    metric = params['metric']
    
  # Now perform the training on full train data.
  model = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               algorithm=algorithm,
                               metric=metric,
                               n_jobs=num_jobs)
  model = model.fit(train_features, train_labels)
  
  return model
  

def train_random_forest(train_features,
                        train_labels,
                        penalty_weights,
                        skip_grid_search,
                        evaluation,
                        num_jobs,
                        num_trees,
                        criterion,
                        max_features,
                        min_samples_split,
                        min_samples_leaf):
  """
  returns the trained random forest model. num_trees, criterion, max_features,
  min_samples_split and min_samples_leaf are ignored if grid search is requested.
  """
  # requested grid search. find best parameters, to achieve highest average score
  if not skip_grid_search:
    algorithm = "random-forest"
    clf = grid_search.grid_search(evaluation,
                                  train_features,
                                  train_labels,
                                  penalty_weights,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    num_trees = max(params['n_estimators'], num_trees)
    criterion = params['criterion']
    max_features = params['max_features']
    min_samples_split = params['min_samples_split']
    min_samples_leaf = params['min_samples_leaf']

  # Now perform the training on full train data.
  model = RandomForestClassifier(n_estimators=num_trees,
                                 n_jobs=num_jobs,
                                 criterion=criterion,
                                 max_features=max_features,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 class_weight=penalty_weights)
  model = model.fit(train_features, train_labels)
  
  return model
  

def train_svm(train_features,
              train_labels,
              penalty_weights,
              skip_grid_search,
              evaluation,
              num_jobs,
              kernel,
              cost,
              gamma,
              degree,
              multi_class):
  """
  returns the trained (kernelized) svm model. cost, gamma, degree and multi_class are
  ignored if grid search is requested.
  Assumes the features are scaled!
  """
  # requested grid search. find best parameters, to achieve highest average score
  if not skip_grid_search:
    algorithm = "linear-svm" if kernel == "linear" else "kernel-svm"
    clf = grid_search.grid_search(evaluation,
                                  train_features,
                                  train_labels,
                                  penalty_weights,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    cost = params['C']
    # multiclass strategy has different attributes in linearsvm and kernel svm
    if kernel == "linear":
      multi_class = params['multi_class']
    else:
      kernel = params['kernel']
      if 'gamma' in params: 
        gamma = params['gamma']
      if 'degree' in params:
        degree = params['degree']
      multi_class = params['decision_function_shape']

  # Now perform the training on full train data.
  if kernel == "linear":
    model = svm.LinearSVC(tol=0.005, max_iter=3000, C=cost,
                          multi_class=multi_class, class_weight=penalty_weights)
  else:
    model = svm.SVC(tol=0.005, cache_size=6000, class_weight=penalty_weights,
                    decision_function_shape=multi_class,
                    kernel = kernel, gamma = gamma, C = cost, degree = degree)
  model = model.fit(train_features, train_labels)
  
  return model


def train_ordinal_logistic(train_features,
                           train_labels,
                           skip_grid_search,
                           evaluation,
                           num_jobs,
                           loss,
                           alpha,
                           cost,
                           ordinal_algorithm):
  """
  returns the trained ordinal logistic model. loss, alpha and cost are ignored if grid
  search is requested.
  alpha: used only for se, it, at, and ridge and if grid search is not requested
  cost: used only for lad and if grid search is not requested
  loss: used only for lad and if grid search is not requested
  """
  # requested grid search. find best parameters, to achieve highest average score
  if not skip_grid_search:
    penalty_weights = 'dummy'
    clf = grid_search.grid_search(evaluation,
                                  train_features,
                                  train_labels,
                                  penalty_weights,
                                  ordinal_algorithm,
                                  num_jobs)
    params = clf.best_params_
    if 'penalty' in params:
      loss = params['loss']
    if 'alpha' in params:
      alpha = params['alpha']
    if 'cost' in params:
      cost = params['cost']

  # Now perform the training on full train data.
  if ordinal_algorithm == 'logisticse':
    model = mord.LogisticSE(alpha = alpha, max_iter=20000)
  elif ordinal_algorithm == 'logisticit':
    model = mord.LogisticIT(alpha = alpha, max_iter=20000)
  elif ordinal_algorithm == 'logisticat':
    model = mord.LogisticAT(alpha = alpha, max_iter=20000)
  elif ordinal_algorithm == 'ordinalridge':
    model = mord.OrdinalRidge(alpha = alpha) 
  elif ordinal_algorithm == 'lad':
    model= mord.LAD(C = cost, loss = loss, max_iter=10000)
  model = model.fit(train_features, train_labels)
  
  return model
  

def train_ridge_classifier(train_features,
                           train_labels,
                           penalty_weights,
                           skip_grid_search,
                           evaluation,
                           num_jobs,
                           alpha,
                           normalize):

  """
  returns the trained ordinal ridge classifier model. loss, alpha and normalize are
  ignored if grid search is requested.
  """
  # requested grid search. find best parameters, to achieve highest average score
  if not skip_grid_search:
    algorithm = "ridgeclassifier"
    clf = grid_search.grid_search(evaluation,
                                  train_features,
                                  train_labels,
                                  penalty_weights,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    alpha = params['alpha']
    normalize = params['normalize']

  # Now perform the training on full train data.
  model = RidgeClassifier(max_iter=20000,
                          alpha=alpha,
                          normalize=normalize,
                          class_weight=penalty_weights)
  model = model.fit(train_features, train_labels)
  
  return model
