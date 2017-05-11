#!/usr/bin/python

# this file includes the methods that train logistic, linear regression and multinomial
# models and return them

import collections
import numpy as np
import utils
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import r2_score

def train_linear(train_features, train_labels, test_features,
                 weight_type,
                 num_jobs):
  """
  Performs the cross validation, and returns the trained model
  """

  model = LinearRegression(normalize = True)
  weights = None
  if weight_type == 'linear':
    weights = np.abs(train_labels - np.mean(train_labels))
  elif weight_type == 'root':
    weights = np.sqrt(np.abs(train_labels - np.mean(train_labels)))
  elif weight_type == 'square':
    weights = np.power(np.abs(train_labels - np.mean(train_labels)), 2)
  elif weight_type == 'freq' or weight_type == 'freq-square':
    age_counts = collections.Counter(train_labels)
    weights_dict = dict()
    max_group_size = max(age_counts.values())
    for (age, count) in age_counts.iteritems():
      weights_dict[age] = 1.0 * max_group_size / count
    if weight_type == 'freq':
      weights = [weights_dict[age] for age in train_labels]
    else:
      weights = [(weights_dict[age])**2 for age in train_labels]
  
  model.fit(train_features, train_labels, sample_weight = weights)
  
  return model
 

def train_elasticnet(train_features, train_labels, test_features,
                     num_alphas,
                     skip_cross_validation, alpha, l1_ratio, 
                     num_jobs):
  """
  Performs the cross validation, and returns the trained model
  """

  if not skip_cross_validation:
    # use 5 fold cross validation
    model = ElasticNetCV(l1_ratio = [0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.9995, 1],
                         max_iter = 30000,
                         cv = 5,
                         n_alphas = num_alphas,
                         n_jobs = num_jobs,
                         normalize = True,
                         tol = 0.005)
  else:
    model = ElasticNet(alpha = alpha,
                       l1_ratio = l1_ratio,
                       normalize = True,
                       max_iter = 30000,
                       tol = 0.005)
  model.fit(train_features, train_labels)
 
  if not skip_cross_validation:
    print("Optimal alpha is {}".format(model.alpha_))
    print("Optimal l1_ratio is {}".format(model.l1_ratio_))
    print("number of iterations were {}".format(model.n_iter_))

  return model


def train_logistic(train_features,
                   train_labels,
                   skip_cross_validation,
                   multi_class,
                   penalty,
                   evaluation,
                   num_jobs,
                   cost):
  """
  returns the trained logistic model with multinomial or ovr functional form. cost is
  ignored if cross validation is requested. penalty is also ignored if mutli_class is
  multinomial as it only works with l2.
  """

  if multi_class == 'ovr':
    solver = 'liblinear'
  elif multi_class == 'multinomial':
    solver = 'lbfgs'
    penalty = 'l2'

  if not skip_cross_validation:
    # use 5 fold cross validation
    model = LogisticRegressionCV(Cs = (10.0**np.arange(-6,5)).tolist(),
                                 class_weight = 'balanced',
                                 cv = 5,
                                 penalty = penalty,
                                 scoring = utils.create_scorer(evaluation),
                                 solver = solver,
                                 tol = 0.0005,
                                 max_iter = 10000,
                                 n_jobs = num_jobs,
                                 refit = True,
                                 multi_class = multi_class)
  else:
    model = LogisticRegression(C = cost,
                               class_weight = 'balanced',
                               penalty = penalty,
                               solver = solver,
                               tol = 0.0005,
                               max_iter = 10000,
                               multi_class = multi_class)

  model.fit(train_features, train_labels)
 
  if not skip_cross_validation:
    print("Optimal cost is {}".format(model.C_))

  return model
