#!/usr/bin/python

import argparse
import pandas
import utils
import models
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import model_selection

parser = argparse.ArgumentParser(
  description='Logistic regression script.')
parser.add_argument('input_filename')
parser.add_argument('-o', '--output_prefix', dest='output_filename_prefix', default=None,
                    help='If provided, the program wil write the test data features '
                    'along with their true/predicted gender to an output file, one per '
                    'gender (one file will be all true females in test data and another '
                    'will be all true males in test data). Default is None which '
                    'disables these outputfiles.')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=94,
                    help='The column number of the label in the input csv. Default is '
                         '94')
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
parser.add_argument('-sb', '--scikit_balancing', dest='scikit_balancing', default=False,
                    action='store_true',
                    help='Whether to use scikit data balancing by changing penalites in '
                    'logistic regression formulation or manually balance by '
                    'undersampling majority class and oversampling minority class.')
parser.add_argument('-tol', '--tolerance', dest='tol', type=float, default=0.0005,
                    help='Tolerance for the stopping criteria.')
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
                    help='Number of jobs to instantiate for grid search. Default is '
                    '-1, which corresponds to the number of cores in the machine')
parser.add_argument('-p', '--penalty', dest='penalty', default='l2',
                    help='The norm to use for the penalty. Must be l1 or l2. Relevent '
                    'only if grid search is skipped.')
parser.add_argument('-d', '--dual', dest='dual', default=False, action='store_true',
                    help='Dual or primal formulation. Dual formulation is only '
                    'implemented for the l2 penalty. Prefer dual=False when '
                    'n_samples > n_features. Relevent only if grid search is skipped.')
parser.add_argument('-c', '--cost', dest='cost', default=1.0, type=float,
                    help='Inverse of the regularization strength; must be a positive '
                    'float. Like in support vector machines, smaller values mean '
                    'stronger regularization. Relevent only if grid search is skipped.')

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
  
  (model, train_features, train_labels, test_features) = models.train_logistic(
      orig_train_features, train_labels, orig_test_features,
      args.scikit_balancing, args.train_size,
      args.skip_feature_selection, args.skip_grid_search,
      args.penalty, args.cost, args.dual, args.tol,
      args.num_jobs)

  # Report accuracy
  y_true, y_pred = test_labels, model.predict(test_features)
  predicted_probabilities = model.predict_proba(test_features)
  print('Test Accuracy: %0.2f%%' % (model.score(test_features, test_labels)*100.))
  print('AUC score: %0.5f' % roc_auc_score(y_true, predicted_probabilities[:,1]))

  # full report
  print("\n*****************************\n")
  label_values = [utils.FEMALE, utils.MALE]
  target_names = ['female', 'male']
  print(classification_report(y_true, y_pred, label_values, target_names))

  # evaluation on test data at different probability thresholds
  utils.print_threshold_metrics(predicted_probabilities, y_true, label_values)

  # write test features along with last bit indicating whether prediction was correct.
  if args.output_filename_prefix:
    utils.write_data_predictions(args.output_filename_prefix, orig_test_features,
                                 feature_names, y_true, y_pred)

if __name__ == '__main__':
  main()
