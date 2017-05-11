import numpy as np
import sys
import utils

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score


def ElasticNetFeatureSelection(train_features, train_labels, test_features, feature_names,
                               l1_ratio, alpha, threshold,
                               num_jobs):
  """
  Performs feature selection, using elastic net, and returns the transformed train/test
  data after feature selection
  """
  if alpha and l1_ratio:
    model = ElasticNet(alpha = alpha,
                       l1_ratio = l1_ratio,
                       normalize = True,
                       max_iter = 30000,
                       tol = 0.005)
  else:
    model = ElasticNetCV(l1_ratio = [0.95, 0.99, 0.995, 0.9995, 1],
                         max_iter = 30000,
                         cv = 3,
                         n_jobs = num_jobs,
                         normalize = True,
                         tol = 0.005)

  feature_selector = SelectFromModel(model, threshold=threshold)
  feature_selector.fit(train_features, train_labels)
  if not alpha or not l1_ratio:
    print("Optimal feature selectoion alpha is {}".format(feature_selector.estimator_.alpha_))
    print("Optimal feature selection l1_ratio is {}".format(feature_selector.estimator_.l1_ratio_))
  train_features = feature_selector.transform(train_features)
  test_features = feature_selector.transform(test_features)
  feature_names = feature_names[feature_selector.get_support(indices=True)]
  print 'Selected {} features '.format(train_features.shape[1])

  return (train_features, test_features, feature_names)


def L1SVMFeatureSelection(train_features, train_labels, test_features, feature_names,
                          feature_selection_cost, threshold,
                          num_jobs):
  """
  Performs feature selection, using lasso svm, and returns the transformed train/test
  data after feature selection
  """
  # features have to be scaled for svm
  (scaled_train_features, scaled_test_features) = utils.scale_data(
      train_features, test_features, 'minmax')

  if feature_selection_cost:
    model = LinearSVC(C = feature_selection_cost,
                      dual = False,
                      penalty = 'l1',
                      tol = 0.005,
                      multi_class = 'ovr',
                      max_iter = 50000)
  else:
    tuned_parameters = [{'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                               1, 5, 10, 50, 100, 500, 1000]}]
    feature_selector_model = LinearSVC(penalty = 'l1',
                                       dual = False,
                                       tol = 0.005,
                                       multi_class = 'ovr',
                                       max_iter = 50000)
    scorer = make_scorer(precision_score, pos_label = None, average = 'macro')
    # Don't pre dispatch all jobs at once, only dispatch ones you are runnings so memory
    # usage does not blow up
    skf = StratifiedKFold(n_splits=3, shuffle = True)
    model = GridSearchCV(estimator=feature_selector_model,
                         param_grid=tuned_parameters,
                         n_jobs=num_jobs,
                         pre_dispatch="n_jobs",
                         cv=skf,
                         scoring=scorer)
 
  model.fit(scaled_train_features, train_labels)
  feature_selector = SelectFromModel(model.best_estimator_, threshold=threshold,
                                     prefit=True)
  if not feature_selection_cost:
    print("Optimal L1-SVM feature selectoion params {}".format(model.best_params_)) 
  train_features = feature_selector.transform(train_features)
  test_features = feature_selector.transform(test_features)
  feature_names = feature_names[feature_selector.get_support(indices=True)]
  print 'Selected {} features '.format(train_features.shape[1])

  return (train_features, test_features, feature_names)
