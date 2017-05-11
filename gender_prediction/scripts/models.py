#!/usr/bin/python

# this file includes the methods that train svm, rf, logistic and knn models and return
# them

import numpy
import utils
import grid_search
import feature_selection
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_svm(train_features, train_labels, test_features,
              scikit_balancing, train_size,
              scaling_method, minmax_min, minmax_max,
              skip_feature_selection, skip_grid_search,
              kernel, gamma, cost, degree,
              num_jobs):
  """ Balances, extracts the requested train size, imputes, scales and finally performs
  features selection on the train data. Then it performs grid search, train a model using
  the best parameters.

  Performs all the data transformations on test data and returns the trained model and the
  transformed test data
  """
  # balance the train data set and create requested train size.
  train_features, train_labels, penalty_weights = utils.prepare_train_data(
      train_features, train_labels, scikit_balancing, train_size)
  
  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(train_features)
  train_features = imputer.transform(train_features)
  test_features = imputer.transform(test_features)

  # now that we have limited the data to requested train size, scale data
  (train_features, test_features) = utils.scale_data(train_features, test_features,
                                                     scaling_method,
                                                     minmax_min, minmax_max)
  
  if not skip_feature_selection:
    feature_selector_obj = feature_selection.feature_selector(train_features,
                                                              train_labels,
                                                              len(train_labels),
                                                              scikit_balancing)
    feature_selector_obj.select_optimal_set(num_jobs)
    train_features = feature_selector_obj.transform(train_features)
    test_features = feature_selector_obj.transform(test_features)
    print("Selected %d features for grid search and final test." %
          len(feature_selector_obj.get_selected_features()))
  
  # requested grid search. find best parameters, to achieve highest average recall
  if not skip_grid_search:
    algorithm = "linear-svm" if kernel == "linear" else "kernel-svm"
    clf = grid_search.grid_search("macro-recall",
                                  train_features,
                                  train_labels,
                                  scikit_balancing,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    print("Best Parameters are: {} ".format(params))
    print("Best Cross Validation Score (mean, std): ({},{})".format(
          clf.cv_results_['mean_test_score'][clf.best_index_],
          clf.cv_results_['std_test_score'][clf.best_index_]))
    if 'kernel' in params:
      kernel = params['kernel']
    if 'gamma' in params:
      gamma = params['gamma']
    if 'C' in params:
      cost = params['C']
    if 'degree' in params:
      degree = params['degree']

  # Now perform the training on full train data. check on test data
  # We enable probability estimates, so that we can identify the top samples.
  model = svm.SVC(tol=0.05, cache_size=6000, class_weight=penalty_weights,
                  kernel = kernel, gamma = gamma, C = cost, degree = degree,
                  probability=True)
  model = model.fit(train_features, train_labels)

  return (model, train_features, train_labels, test_features)


def train_logistic(train_features, train_labels, test_features,
                   scikit_balancing, train_size,
                   skip_feature_selection, skip_grid_search,
                   penalty, cost, dual, tol,
                   num_jobs):
  """
  Performs all the data transformations on test data and returns the trained model and the
  transformed test data
  """
  # balance the train data set and create requested train size.
  train_features, train_labels, penalty_weights = utils.prepare_train_data(
      train_features, train_labels, scikit_balancing, train_size)
  
  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(train_features)
  train_features = imputer.transform(train_features)
  test_features = imputer.transform(test_features)

  if not skip_feature_selection:
    # feature selector expects scaled features
    (scaled_train_features, scaled_test_features) = utils.scale_data(
        train_features, test_features, 'minmax')
    feature_selector_obj = feature_selection.feature_selector(scaled_train_features,
                                                              train_labels,
                                                              len(train_labels),
                                                              scikit_balancing)
    feature_selector_obj.select_optimal_set(num_jobs)
    train_features = feature_selector_obj.transform(train_features)
    test_features = feature_selector_obj.transform(test_features)
    print("Selected %d features for grid search and final test." %
          len(feature_selector_obj.get_selected_features()))

  # requested grid search. find best parameters, to achieve highest average recall
  if not skip_grid_search:
    algorithm = "logistic"
    clf = grid_search.grid_search("macro-recall",
                                  train_features,
                                  train_labels,
                                  scikit_balancing,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    print("Best Parameters are: {} ".format(params))
    print("Best Cross Validation Score (mean, std): ({},{})".format(
          clf.cv_results_['mean_test_score'][clf.best_index_],
          clf.cv_results_['std_test_score'][clf.best_index_]))
    penalty = params['penalty']
    cost = params['C']

  # Now perform the training on full train data. check on test data
  model = LogisticRegression(penalty=penalty,
                             dual=dual,
                             C=cost,
                             tol=tol,
                             max_iter=5000,
                             class_weight=penalty_weights)
  model = model.fit(train_features, train_labels)
  
  return (model, train_features, train_labels, test_features)


def train_random_forest(train_features, train_labels, test_features,
                        scikit_balancing, train_size,
                        skip_feature_selection, skip_grid_search,
                        max_features, n_estimators, criterion, min_samples_split,
                        min_samples_leaf, num_jobs):
  """
  Performs all the data transformations on test data and returns the trained model and the
  transformed test data
  """
  # balance the train data set and create requested train size.
  train_features, train_labels, penalty_weights = utils.prepare_train_data(
      train_features, train_labels, scikit_balancing, train_size)
  
  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(train_features)
  train_features = imputer.transform(train_features)
  test_features = imputer.transform(test_features)

  if not skip_feature_selection:
    # feature selector expects scaled features
    (scaled_train_features, scaled_test_features) = utils.scale_data(
        train_features, test_features, 'minmax')
    feature_selector_obj = feature_selection.feature_selector(scaled_train_features,
                                                              train_labels,
                                                              len(train_labels),
                                                              scikit_balancing)
    feature_selector_obj.select_optimal_set(num_jobs)
    train_features = feature_selector_obj.transform(train_features)
    test_features = feature_selector_obj.transform(test_features)
    print("Selected %d features for grid search and final test." %
          len(feature_selector_obj.get_selected_features()))

  max_features = utils.extract_max_features(max_features)
  # requested grid search. find best parameters, to achieve highest average recall
  if not skip_grid_search:
    algorithm = "random-forest"
    clf = grid_search.grid_search("macro-recall",
                                  train_features,
                                  train_labels,
                                  scikit_balancing,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    print("Best Parameters are: {} ".format(params))
    print("Best Cross Validation Score (mean, std): ({},{})".format(
          clf.cv_results_['mean_test_score'][clf.best_index_],
          clf.cv_results_['std_test_score'][clf.best_index_]))
    n_estimators = max(params['n_estimators'], n_estimators)
    criterion = params['criterion']
    max_features = params['max_features']
    min_samples_split = params['min_samples_split']
    min_samples_leaf = params['min_samples_leaf']

  # Now perform the training on full train data. check on test data
  model = RandomForestClassifier(n_estimators=n_estimators,
                                 n_jobs=num_jobs,
                                 criterion=criterion,
                                 max_features=max_features,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 class_weight=penalty_weights)
  model = model.fit(train_features, train_labels)
  
  return (model, train_features, train_labels, test_features)

def train_knn(train_features, train_labels, test_features,
              imbalanced_data, train_size,
              scaling_method, minmax_min, minmax_max,
              skip_feature_selection, skip_grid_search,
              n_neighbors, weights, algorithm, metric,
              num_jobs):
  """
  Performs all the data transformations on test data and returns the trained model and the
  transformed test data
  """
  # balance the train data set and create requested train size. Here instead of
  # scikit balancing, we will use imbalanced_data flag and discard the last output since
  # it is irrelevant to knn. In order not to balance the data, the third argument should
  # be true (simulate scikit balancing); so we will use imabalanced_data flag in place of
  # scikit_balancing.
  train_features, train_labels, dummy = utils.prepare_train_data(
      train_features, train_labels, imbalanced_data, train_size)
  
  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(train_features)
  train_features = imputer.transform(train_features)
  test_features = imputer.transform(test_features)

  # now that we have limited the data to requested train size, scale data
  (train_features, test_features) = utils.scale_data(train_features, test_features,
                                                     scaling_method,
                                                     minmax_min, minmax_max)
  
  if not skip_feature_selection:
    feature_selector_obj = feature_selection.feature_selector(train_features,
                                                              train_labels,
                                                              len(train_labels),
                                                              imbalanced_data)
    feature_selector_obj.select_optimal_set(num_jobs)
    train_features = feature_selector_obj.transform(train_features)
    test_features = feature_selector_obj.transform(test_features)
    print("Selected %d features for grid search and final test." %
          len(feature_selector_obj.get_selected_features()))
  
  # requested grid search. find best parameters, to achieve highest average recall
  if not skip_grid_search:
    algorithm = "knn"
    clf = grid_search.grid_search("macro-recall",
                                  train_features,
                                  train_labels,
                                  imbalanced_data,
                                  algorithm,
                                  num_jobs)
    params = clf.best_params_
    print("Best Parameters are: {} ".format(params))
    print("Best Cross Validation Score (mean, std): ({},{})".format(
          clf.cv_results_['mean_test_score'][clf.best_index_],
          clf.cv_results_['std_test_score'][clf.best_index_]))
    n_neighbors = params['n_neighbors']
    weights = params['weights']
    algorithm = params['algorithm']
    metric = params['metric']
    
  # Now perform the training on full train data. check on test data
  model = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               algorithm=algorithm,
                               metric=metric)
  model = model.fit(train_features, train_labels)
  
  return (model, train_features, train_labels, test_features)
