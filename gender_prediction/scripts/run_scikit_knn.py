#!/usr/bin/python

import argparse
import pandas
import utils
import models
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import model_selection

parser = argparse.ArgumentParser(
    description='This script runs KNN algorithm on the scaled data.')
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
parser.add_argument('-s', '--scaler', dest='scaling_method',
                    choices=['normal', 'standard', 'minmax'], default="minmax",
                    help='The type of scaling method to use. Possible options are: '
                    'normal, standard, minmax. '
                    'normal: normalizes the features based on L2 norm. '
                    'standard: standardizes features so that they are zero centered and '
                    'have unit variance. '
                    'minmax: scales the features based on minimum and maximum so that '
                    'they are between 0 and 1.')
parser.add_argument('-lb', '--min', dest='minmax_min', type=float, default=0,
                    help='The scaling range minimum if scaling method is minmax')
parser.add_argument('-ub', '--max', dest='minmax_max', type=float, default=1,
                    help='The scaling range maximum if scaling method is minmax')
parser.add_argument('-i', '--imbalanced_data', dest='imbalanced_data',
                    default=False, action='store_true',
                    help='Whether to balance training data or not by including equal '
                    'number of positive and negative labels in training set. Default is '
                    'to balance the data.')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=94,
                    help='The column number of the label in the input csv. Defautl is '
                    '94, set it otherwise')
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
parser.add_argument('-nn', '--num_neighbors', dest='num_neighbors',
                    default=5, type=int,
                    help='The number of nearest neighbors to use with KNN. Relevent only '
                    'if grid search is skipped.')
parser.add_argument('-w', '--weights', dest='weights',
                    default='distance', choices=['uniform', 'distance'],
                    help='The scheme according to which weights are given to neighbors. '
                    'Default is distance. Relevent only if grid search is skipped.')
parser.add_argument('-a', '--algorithm', dest='algorithm',
                    default='auto', choices=['auto', 'ball_tree', 'kd_tree', 'brute'],
                    help='Algorithm used to find nearest neighbors. Default is auto. '
                    'Relevent only if grid search is skipped.')
parser.add_argument('-m', '--metric', dest='metric',
                    default='euclidean',
                    choices=['euclidean', 'manhattan', 'chebyshev'],
                    help='The distance metric to use for evaluating neighbors. Default '
                    'is euclidean. Relevent only if grid search is skipped.')
args = parser.parse_args()

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

  (model, train_features, train_labels, test_features) = models.train_knn(
      orig_train_features, train_labels, orig_test_features,
      args.imbalanced_data, args.train_size,
      args.scaling_method, args.minmax_min, args.minmax_max,
      args.skip_feature_selection, args.skip_grid_search,
      args.num_neighbors, args.weights, args.algorithm, args.metric,
      args.num_jobs)
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
  
  # write test features along with last bit indicating whether prediction was correct.
  if args.output_filename_prefix:
    utils.write_data_predictions(args.output_filename_prefix, orig_test_features,
                                 feature_names, y_true, y_pred)

if __name__ == "__main__":
  main()
