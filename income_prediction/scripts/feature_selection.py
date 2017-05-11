#!/usr/bin/python

import csv
import pandas
import numpy
import sys
import utils
import warnings

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.linear_model import LogisticRegression

class feature_selector(object):
  """ An class for extracting top featueres from input data.
  """
  def __init__(self,
               evaluation,
               all_features,
               all_labels,
               feature_label_names,
               num_samples,
               penalty_weights,
               algorithm,
               num_jobs):

    '''
    Initializes a feature selector object by cross validating the selector model and
    storing the best transformer.

    all_feature: Must contain the features all scaled to the same range if l1-svm feature
    seleciton if requested.
    all_labels: labels corresponding to features
    num_samples: how many random data points to sample and use from scaled_features for
    penalty_weights: determines how to account for imbalance. either None, which means the
    data is balanced, or balanced which means use scikit balacning.
    training the feature selector models, to speed up the whole process.
    algorithm: what feature selection method to use: anova, logistic, tree
    '''
    # copy the features and labels, because we don't want to be modifying (scaling or
    # sampling) in place
    features = numpy.copy(all_features)
    labels = numpy.copy(all_labels)
    self.features_label_names_ = feature_label_names

    # scale features?
    if False:
      (feautres, dummy) = utils.scale_data(features, None, 'standard')
    
    if num_samples > 0:
      # select a smaller sample for feature selection
      indices = numpy.random.choice(features.shape[0], num_samples, replace=False)
      features = features[indices, :]
      labels = labels[indices]

    # Set the parameters for gid search and model based on algorithm choice
    if algorithm == 'anova' or algorithm == 'best':
      self.perform_feature_selection(evaluation, features, labels, penalty_weights,
                                     algorithm, num_jobs)
    else:
      sys.exit('bad algorithm for feature selection: ' + algorithm)
   
    self.best_params_ = self.clf_.best_params_
    print "Best Feature Selection Parameters are: " + str(self.best_params_)
    print "Best Feature Selection CV Score: " + str(self.clf_.best_score_)
    
    best_score_func = self.best_params_['feature_selection__score_func']
    best_percentile = self.best_params_['feature_selection__percentile']
    self.best_feature_selector_ = self.clf_.best_estimator_.named_steps['feature_selection']
  
  def perform_feature_selection(self,
                                evaluation,
                                features,
                                labels,
                                penalty_weights,
                                algorithm,
                                num_jobs):
    # we use best features by their percentile in the sorted importance list and logistic
    # regression for the final evaluation estimator. It could be anything (like svm), but
    # we choose LR since it's fast.
    feature_selector = SelectPercentile() 
    logistic_model = LogisticRegression(max_iter=20000, class_weight=penalty_weights)
    steps = [('feature_selection', feature_selector),
             ('logistic', logistic_model)]
    pipeline = Pipeline(steps)
    score_funcs = list()
    if algorithm == 'best' or algorithm == 'anova':
      score_funcs.extend([f_classif, f_regression])
    # Parameters of the estimators in the pipeline can be accessed using the
    # <estimator>__<parameter> syntax. feature selection criteria are f_classif.
    # We also add f test for regression, since the dependent variable is an ordinal
    # variable, not simply a categorical variable
    # we don't search multionomial logistic regression here, because this is just feature
    # selection. If we get good results with ovr multi_class, most likely we will also get
    # it with multinomial. Also in previous experiments ovr was outperforming multinomial.
    tuned_parameters = [
        {'feature_selection__percentile': [2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100],
         'feature_selection__score_func': score_funcs,
         'logistic__multi_class': ['ovr'],
         'logistic__solver': ['liblinear'],
         'logistic__penalty': ['l1', 'l2'],
         'logistic__C': [0.0001, 0.01, 1, 100, 10000]}]
    skf = model_selection.StratifiedKFold(labels, n_folds=5, shuffle = True)
    scorer = utils.create_scorer(evaluation)
    # Don't pre dispatch all jobs at once, only dispatch ones you are runnings so memory
    # usage does not blow up
    clf = GridSearchCV(estimator=pipeline,
                       param_grid=tuned_parameters,
                       n_jobs=num_jobs,
                       pre_dispatch="n_jobs",
                       cv=skf,
                       scoring=scorer)
    self.clf_ = clf.fit(features, labels)

   
  def transform(self, original_features):
    ''' Transforms the features into only the selected ones by the best model.

    Return the selected features.
    '''
    return self.best_feature_selector_.transform(original_features)
  
  def get_selected_features(self):
    ''' Return list of all selected features, those that resulted in the best score.
    '''
    selected_feature_indices = self.best_feature_selector_.get_support(indices=True)
    selected_features = [self.features_label_names_[i] for i in selected_feature_indices]
    return selected_features

  
  def get_top_features(self, num_features):
    ''' Return list of top num_features from selected features. top features are those
    that achieved the highest score. if num_features is greater than the number of
    selected features, it will show all.
    '''
    # get a copy of coefficients and sort
    scores = self.best_feature_selector_.scores_.tolist()
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [self.features_label_names_[i] for i in top_indices[:num_features]]
