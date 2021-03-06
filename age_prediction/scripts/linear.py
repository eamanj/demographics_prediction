#!/usr/bin/python

import argparse
import numpy as np
import utils
import models
import feature_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(
  description='linear regression script.')
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
parser.add_argument("-w", "--weight_type", dest="weight_type",
                    default="uniform", choices=["uniform", "square", "linear", "freq",
                                                "freq-square"],
                    help="If provided, determines how samples should be weighted in the "
                    "regression. Choices are linear and square, which use raw and square "
                    "root of difference from mean age as sample weights. freq and "
                    "freq-square also assign weights proportional to the inverse and "
                    "sqaure of inverse of frequency of the age in the population. "
                    "Default is uniform which sets equal weight to all samples.")
parser.add_argument('-c', '--calibrate', dest='calibrate',
                    default=False, action='store_true',
                    help='Should we calibrate the predictions so the predicted ranges '
                    'match on the train data?')
parser.add_argument('-nj', '--num_jobs', dest='num_jobs', type=int, default=-1,
                    help='Number of jobs to instantiate for grid search. Default is '
                    '-1, which corresponds to the number of cores in the machine')

# feature selection flags
parser.add_argument('-sf', '--skip_feature_selection', dest='skip_feature_selection',
                    default=False, action='store_true',
                    help='If specified, skips feature selection. Default is to run '
                    'cross validated feature selection on original features and '
                    'transform data into a smaller dimension.')
parser.add_argument('-l', '--l1_ratio', dest='l1_ratio', default=None, type=float,
                    help='The Elastic Net mixing parameter, must be a float between 0 '
                    'and 1, with 0 <= l1_ratio <= 1. It determines how aggressive is the '
                    'feature selection in elastic net. If not provided, and feature '
                    'feature_selection is true, then it will be determined by cross '
                    'validation.')
parser.add_argument('-a', '--alpha', dest='alpha', default=None, type=float,
                    help='alpha multiplies the penalty terms and thus determines '
                    'the level of regularization. Defaults to 1.0. alpha = 0 is '
                    'equivalent to an ordinary least square. If not provided and '
                    'feature_selection is true, then it will be determined by cross '
                    'validation.')
parser.add_argument('-t', '--threshold', dest='threshold', default=0, type=float,
                    help='Feature selection threshold to elastic net. If not specified '
                    'all features with non-zero importance by elastic net will be used.')
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

  if not args.skip_feature_selection:
    (train_features, test_features, feature_names) = feature_selection.ElasticNetFeatureSelection(
        train_features, train_labels, test_features, feature_names,
        args.l1_ratio, args.alpha, args.threshold,
        args.num_jobs)
 
  model = models.train_linear(train_features, train_labels, test_features,
                              args.weight_type,
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
