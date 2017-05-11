#!/usr/bin/python

# This file cross validates a random forest with the parameters as passed by arguments,
# reports the average cross validation score across several random stratified folds of the
# data. The number of cross validation rounds can be set by an input flag.
# It then print the feature ranking based on the random forest, evaluates the random
# forest accuracy on the test data set and reports the accuracy. Different evaluation
# metrics is later reported on poritions of the test data for which the predicted label
# probability exceeds a thrshold.

import csv
import argparse
import numpy
import pandas
import utils
import models
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import model_selection

parser = argparse.ArgumentParser(
    description='This script performs the cross validation and testing of European data '
                 'using SciKit Random Forest. The testing is performed in two different '
                 'ways. First an aggregate accuracy on all of the data is reported. '
                 'Second, accuracy, precision and recalls are computed on portions of '
                 'the data whose mean probability of the predicted label across all '
                 'trees exceeds a threshold. The fraction of such data points (whose '
                 'predictions we are more confident about) is reported along side the '
                 'evaluation metrics. The thresholds are pre-set and range from 50% ('
                 'which is equivalent to prediction on all points) to 90% (which is the '
                 'most selective threshold on most confident prediction). The '
                 'probabilities returned by a single tree are the normalized class '
                 'histograms of the leaf a sample lands in.')
parser.add_argument('input_filename')
parser.add_argument('-o', '--output_prefix', dest='output_filename_prefix', default=None,
                    help='If provided, the program wil write the test data features '
                    'along with their true/predicted gender to an output file, one per '
                    'gender (one file will be all true females in test data and another '
                    'will be all true males in test data). Default is None which '
                    'disables these outputfiles.')
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
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=94,
                    help='The column number of the label in the input csv. Defautl is '
                    '94, set it otherwise')
parser.add_argument('-sb', '--scikit_balancing', dest='scikit_balancing',
                    default=False, action='store_true',
                    help='Whether to use scikit data balancing by changing sample '
                    'weights or manually balance by undersampling majority class and '
                    'oversampling minority class')
parser.add_argument('-sf', '--skip_feature_selection', dest='skip_feature_selection',
                    default=False, action='store_true',
                    help='If specified, skips feature selection. Default is to run '
                    'cross validated feature selection on original features and '
                    'transform data into a smaller dimension.')
parser.add_argument('-sg', '--skip_grid_search', dest='skip_grid_search',
                    default=False, action='store_true',
                    help='If specified, skips grid search. Default is to run cross '
                    'validation with grid search to determine the best parameters. Then '
                    'run the training once more with best params. If specified, this '
                    'step is skipped, and the provided parameters are used to train the '
                    'model. Default is False which performs the grid search.')
parser.add_argument('-nj', '--num_jobs', dest='num_jobs', type=int, default=-1,
                    help='Number of jobs to instantiate for grid search and training '
                    'trees. Default is -1, which corresponds to the number of cores in '
                    'the machine')
parser.add_argument('-nt', '--num_trees', dest='num_trees', type=int, default=400,
                    help = 'The number of trees in the forest. Relevent only if grid '
                    'search is skipped or this is larger than grid search num trees.')
parser.add_argument('-c', '--criterion', dest='criterion', default='gini',
                    choices=['gini', 'entropy'],
                    help='The criterion used for evaluating the trees. Possible values '
                    'are "gini" and "entropy". Relevent only if grid search is skipped.')
parser.add_argument('-mf', '--max_features', dest='max_features', default="auto",
                    help='The number of features to consider when looking for the best '
                    'split. Possible values are: "auto", "sqrt", "log2", "all", a '
                    'float between 0 and 1 or an integer. "all" uses all features, a '
                    'float determines the fraction of features to use and an integer '
                    'explicitly indicated the number of features to use. Relevent only '
                    'if grid search is skipped.')
parser.add_argument('-mss', '--min_samples_split', dest='min_samples_split',
                    default=2, type=int,
                    help='The minimum number of samples required to split an internal '
                    'node. Must be a non-negative integer greater than min_samples_leaf. '
                    'Relevent only if grid search is skipped.')
parser.add_argument('-msl', '--min_samples_leaf', dest='min_samples_leaf',
                    default=1, type=int,
                    help='The minimum number of samples that must exist in a leaf for a '
                    'split to occur. Must be a non-negative integer. Relevent only if '
                    'grid search is skipped.')
args = parser.parse_args()

def print_feature_importances(model):
  with open(args.input_filename) as f:
    reader = csv.reader(f)
    columns = next(reader)
  importances = model.feature_importances_
  indices = numpy.argsort(importances)[::-1]
  print("Feature ranking:")
  for f in range(20):
    print("%d. %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))


def main():
  df = pandas.read_csv(args.input_filename, index_col=False, header=0)
  data = df.values
  column_names = df.columns.values.tolist()
  feature_names = column_names[0:args.label_column]
  label_name = column_names[args.label_column]

  # Extract features/labels and their names from raw data
  features = data[:, 0:args.label_column]
  labels = data[:, args.label_column].astype(int)
  orig_train_features, orig_test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=args.test_size))
  
  (model, train_features, train_labels, test_features) = models.train_random_forest(
      orig_train_features, train_labels, orig_test_features,
      args.scikit_balancing, args.train_size,
      args.skip_feature_selection, args.skip_grid_search,
      args.max_features, args.num_trees, args.criterion, args.min_samples_split,
      args.min_samples_leaf, args.num_jobs)
  # Report accuracy
  y_true, y_pred = test_labels, model.predict(test_features)
  predicted_probabilities = model.predict_proba(test_features)
  print("Test Accuracy: %0.2f%%" % (model.score(test_features, test_labels)*100.))
  print('AUC score: %0.5f' % roc_auc_score(y_true, predicted_probabilities[:,1]))

  # full report
  print("\n*****************************\n")
  labels = [0 , 1]
  target_names = ["female" , "male"]
  print(classification_report(y_true, y_pred, labels, target_names))

  # Now perform the evaluation on the test data at different probability thresholds.
  # The idea is we report the accuracy only for points whose predicted probability
  # for either label is above the specified threshold.
  utils.print_threshold_metrics(predicted_probabilities, y_true, labels)
  
  print("\n")
  print_feature_importances(model)
  
  # write test features along with last bit indicating whether prediction was correct.
  if args.output_filename_prefix:
    utils.write_data_predictions(args.output_filename_prefix, orig_test_features,
                                 feature_names, y_true, y_pred)

if __name__ == "__main__":
  main()

