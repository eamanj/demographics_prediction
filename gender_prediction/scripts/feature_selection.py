#!/usr/bin/python

import csv
import pandas
import numpy
import sys
import utils
import warnings

from sklearn import preprocessing, feature_selection
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

class feature_selector(object):
  """ An class for extracting top featueres from input data.
  """
  def __init__(self, scaled_features, labels, num_samples, scikit_balancing):
    """
    scaled_feature: Must contain the features all scaled to the same range.
    labels: labels corresponding to scaled_features
    num_samples: how many random data points to sample and use from scaled_features for
    training the feature selector models
    """
    # select a smaller sample for feature selection
    indices = numpy.random.choice(scaled_features.shape[0], num_samples, replace=False)
    l1_svm_features = scaled_features[indices, :]
    l1_svm_labels = labels[indices]

    # Manually balance data. Do this on the whole data, because we are training the
    # feature selction on all of data.
    self.features, self.labels, self.penalty_weights = utils.prepare_train_data(
        l1_svm_features, l1_svm_labels, scikit_balancing, -1)

    # a dictionary from svm cost to trained selector model
    self.feature_selector_models = dict()
    
    # Keeps track of the latest model trained for feature selection. All calls for data
    # transformation or feature coefficients are done based on this trained model.
    self.current_model = None
    self.current_transformer = None

  def select_optimal_set(self, num_jobs):
    ''' Finds the best set of features to use through cross validation.

    It performs a grid search cross validation through possible cost values that yields
    the highest performance. The selector object will be set to this optimal model.
    '''
    tuned_parameters = [{'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                               1, 5, 10, 50, 100, 500, 1000]}]
    feature_selector_model = LinearSVC(penalty = 'l1',
                                       dual = False,
                                       max_iter=5000,
                                       tol = 0.005,
                                       class_weight = self.penalty_weights)
    scorer = make_scorer(precision_score, pos_label = None, average = 'macro')
    # Don't pre dispatch all jobs at once, only dispatch ones you are runnings so memory
    # usage does not blow up
    skf = StratifiedKFold(n_splits=5, shuffle = True)
    clf = GridSearchCV(estimator=feature_selector_model,
                       param_grid=tuned_parameters,
                       n_jobs=num_jobs,
                       pre_dispatch="n_jobs",
                       cv=skf,
                       scoring=scorer)
    clf.fit(self.features, self.labels)
    self.current_model = clf.best_estimator_
    self.current_transformer = feature_selection.SelectFromModel(clf.best_estimator_,
                                                                 prefit=True)

  def select_top_features(self, num_features):
    """ Finds the right regularization parameter in L1-SVM that yields num_features to be
    selected and returns the fitted L1-svm selector model.

    num_features: the desired number of features to be selected.

    Finds the right model with num_features coefficients above zero, unless the binary
    search goes on for more than max_num_bisections.
    It sets current_model and current_transformer once done with the model and its feature
    selector.
    """
    # The linear svm cost parameter controls the sparsity of selected features: the
    # smaller, the fewer selected features selected. So if you want aggressive feature
    # selection, reduce this cost parameter.
    # The minimum and maximum values in the binary search to find the right C which yields
    # top num_features.
    min_linear_svm_cost = 0.000001
    max_linear_svm_cost = 500
    max_num_bisections = 200
    num_bisections = 0

    current_num_features = None
    while (current_num_features != num_features and
           num_bisections < max_num_bisections):
      linear_svm_cost = (min_linear_svm_cost + max_linear_svm_cost)/2.0
      num_bisections += 1
      # Perform the feature selection on all of data. Use a linear SVM with L1 penalty
      # which yields a sparse solution
      print ('Trying cost ' + str(linear_svm_cost) + ' with current num_features ' +
             str(current_num_features))
      
      # if we have already trained a model for this svm cost, avoid retraining it.
      if linear_svm_cost in self.feature_selector_models:
        feature_selector_model = self.feature_selector_models[linear_svm_cost]
      else:
        feature_selector_model = LinearSVC(C = linear_svm_cost,
                                           penalty = 'l1',
                                           dual = False,
                                           tol = 0.005,
                                           class_weight = self.penalty_weights)
        feature_selector_model.fit(self.features, self.labels)
        # keep track of this model, so we don't have to retrain it.
        self.feature_selector_models[linear_svm_cost] = feature_selector_model

      transformer = feature_selection.SelectFromModel(feature_selector_model,
                                                      prefit=True)
      current_num_features = sum(transformer.get_support(indices=False))

      if current_num_features < num_features:
        min_linear_svm_cost = linear_svm_cost
      elif current_num_features > num_features:
        max_linear_svm_cost = linear_svm_cost

    if num_bisections >= max_num_bisections:
      print ('Warning: Could not find the matching regularization parameter in ' +
             str(num_bisections) + ' bisections.')
             
    self.current_model = feature_selector_model
    self.current_transformer = transformer
  
  def get_selected_features(self, feature_names=None):
    ''' Returns the list of selected features names and their coefficients from the
    current trained model.
    curent_model should be set by either select_top_features or select_optimal_set, before
    calling this method.
    feature_names: an array containing the list of all feature names as they appear in
    sclaed features. If not provided, it will return the index of selected feature
    '''
    selected_indices = self.current_transformer.get_support(indices=True)
    coefficients = self.current_model.coef_[0]
    output = []
    for index in selected_indices:
      if feature_names:
        output.append((feature_names[index], coefficients[index]))
      else:
        output.append((index, coefficients[index]))
    
    
    # But sort the features according to their absolute value. First get the indices to
    # the sorted scores.
    sorted_output = sorted(output, key = lambda k: abs(k[1]), reverse = True)
    return sorted_output
  
  def transform(self, original_features):
    ''' Transforms the features into only the selected ones by the current model.

    Return the selected features.
    '''
    return self.current_transformer.transform(original_features)
