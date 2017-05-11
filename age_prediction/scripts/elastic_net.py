#!/usr/bin/python

import argparse
import numpy as np
import utils
import models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(
  description='Elastic net script.')
parser.add_argument('input_filename')
parser.add_argument('output_filename')
parser.add_argument('-trs', '--train_size', dest='train_size',
                    type=int, default=-1,
                    help='The size of train set. It must be a whole integer, greater '
                    'than 1 which represents the absolute number of data points to use '
                    'for training. Default is -1 which means all available data after '
                    'splitting into test and train is used. There will be some '
                    'undersampling of the majority class, if train data needs to be '
                    'balanced.')
parser.add_argument('-tes', '--test_size', dest='test_size',
                    type=int, default=15000,
                    help='The size of test set. It must be a whole integer, greater '
                    'than 1 which represents the absolute number of data points to use '
                    'for testing. You need to make sure test and train size do not '
                    'exceed the total size and there are enough samples of each class '
                    'in case train data is balanced.')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=1384,
                    help='The column number of the label in the input csv. Default is '
                    '1384 starting from zero, set it otherwise')
parser.add_argument('-nj', '--num_jobs', dest='num_jobs', type=int, default=-1,
                    help='Number of jobs to instantiate for grid search. Default is '
                    '-1, which corresponds to the number of cores in the machine')
parser.add_argument('-c', '--calibrate', dest='calibrate',
                    default=False, action='store_true',
                    help='Should we calibrate the predictions so the predicted ranges '
                    'match on the train data?')

# cross validation flags: they will be ignored if cross validation is not skipped.
parser.add_argument('-sc', '--skip_cross_validation', dest='skip_cross_validation',
                    default=False, action='store_true',
                    help='If specified, skips cross validation and uses the parameters '
                    'given below. ')
parser.add_argument('-na', '--num_alphas', dest='num_alphas', type=int, default=100,
                    help='The number of alphas to try for each l1 ratio, in case '
                    'cross validation is not skipped. Defaults to 100.')
parser.add_argument('-l', '--l1_ratio', dest='l1_ratio', default=0.15, type=float,
                    help='The Elastic Net mixing parameter, must be a float between 0 '
                    'and 1, with 0 <= l1_ratio <= 1. It determines how aggressive is the '
                    'feature selection in elastic net. Must be specified only if penatly '
                    'is elasticnet, as default (0.15) might not be good.  Relevent only '
                    'if cross validation is skipped.')
parser.add_argument('-a', '--alpha', dest='alpha', default=1, type=float,
                    help='alpha multiplies the penalty terms and thus determines '
                    'the level of regularization. Defaults to 1.0. alpha = 0 is '
                    'equivalent to an ordinary least square.')
args = parser.parse_args()


def main():
  add_log_vars = True
  (train_features, train_labels, test_features, test_labels, class_values,
   feature_names, label_name) = utils.prepare_data(args.input_filename,
                                                   args.label_column,
                                                   args.train_size,
                                                   args.test_size,
                                                   add_log_vars)

  max_label = max(class_values)
  min_label = min(class_values)
  print("Max label value: %d" % max_label)
  print("Min label value: %d" % min_label)
  print("Label {}".format(label_name))

  model = models.train_elasticnet(train_features, train_labels, test_features,
                                  args.num_alphas, 
                                  args.skip_cross_validation, args.alpha, args.l1_ratio, 
                                  args.num_jobs)
  
  
  # Predict on train/test
  train_y_true, train_y_pred = train_labels, model.predict(train_features)
  test_y_true, test_y_pred = test_labels, model.predict(test_features)


  if args.calibrate:
    # transform predicted range (because we should actually be doing truncated regression)
    # so that predicted ranges match in low/high buckets
    low_bucket_true = train_y_true[train_y_true < 30]
    low_bucket_pred = train_y_pred[train_y_true < 30]
    min_train_true = np.median(low_bucket_true)
    min_train_pred = np.median(low_bucket_pred)
    high_bucket_true = train_y_true[train_y_true > 70]
    high_bucket_pred = train_y_pred[train_y_true > 70]
    max_train_true = np.median(high_bucket_true)
    max_train_pred = np.median(high_bucket_pred)
    m = (max_train_true - min_train_true) * 1.0 / (max_train_pred - min_train_pred)
    b = max_train_true - (m * max_train_pred)
    print 'True max/min train {}/{}'.format(max_train_true, min_train_true)
    print 'Pred max/min train {}/{}'.format(max_train_pred, min_train_pred)
    print 'calibration slope: {}, calibration intercept: {}'.format(m, b)
    train_y_pred = m * train_y_pred + b
    test_y_pred = m * test_y_pred + b
  
  
  print('\n*****************************\n')
  print('R-Squared on train (weighted uniformly) is %f' %
        r2_score(train_y_true, train_y_pred, multioutput = 'uniform_average'))
  print('R-Squared on train (weighted by variance) is %f' %
        r2_score(train_y_true, train_y_pred, multioutput = 'variance_weighted'))
  
  print('\n')
  print('R-Squared on test (weighted uniformly) is %f' %
        r2_score(test_y_true, test_y_pred, multioutput = 'uniform_average'))
  print('R-Squared on test (weighted by variance) is %f' %
        r2_score(test_y_true, test_y_pred, multioutput = 'variance_weighted'))

  print('\n')
  print('MAE on train: %f' %
        mean_absolute_error(train_y_true, train_y_pred, multioutput='uniform_average'))
  print('MSE on train: %f' %
        mean_squared_error(train_y_true, train_y_pred, multioutput='uniform_average'))
  
  print('\n')
  print('MAE on test: %f' %
        mean_absolute_error(test_y_true, test_y_pred, multioutput='uniform_average'))
  print('MSE on test: %f' %
        mean_squared_error(test_y_true, test_y_pred, multioutput='uniform_average'))

  # write the test output to file
  utils.write_data_predictions(args.output_filename,
                               test_y_true,
                               test_y_pred)

if __name__ == '__main__':
  main()
