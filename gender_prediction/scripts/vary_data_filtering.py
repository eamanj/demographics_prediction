#!/usr/bin/python
import argparse
import pandas
import utils
import csv
import math
import random
import sys
import multiprocessing
import models
from numpy import mean, random, arange
from collections import defaultdict

from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score

parser = argparse.ArgumentParser(
    description='This script test prediction on a fixed test size at different levels '
                'of data filtering. Data filtering is done based on number of active '
                'days per week on average. The predictions stats are recorded as data '
                'is filtered more aggresively until requiring at least 6 days of '
                'activity.')
parser.add_argument('-trs', '--train_size', dest='train_size',
                    type=int, default = 15000,
                    help='The size of train set. It must be a whole integer, greater '
                    'than 1 which represents the absolute number of data points to use '
                    'for training. Default is -1 which means all available data after '
                    'splitting into test and train is used. There will be some '
                    'undersampling of the majority class, if train data needs to be '
                    'balanced.')
parser.add_argument('-tes', '--test_size', dest='test_size',
                    type=int, default = 15000,
                    help='The number of data points to keep aside as test. '
                    'The size of test remains fixed, even though train size changes. '
                    'The default value is 20000 data points from original data. Provide '
                    'an integer less than total size of the original data. Also make '
                    'sure you leave a good proportion of the data for training.')
parser.add_argument('-la', '--learning_algorithm', dest='learning_algorithm',
                    choices=['svm', 'random-forest', 'logistic', 'knn'],
                    default='svm',
                    help='Determines the learning algorithm to employ')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=94,
                    help='The column number of the label in the input csv. Defautl is '
                    '94, set it otherwise')
parser.add_argument('-sb', '--scikit_balancing', dest='scikit_balancing',
                    default=False, action='store_true',
                    help='Whether to use scikit data balancing by changing sample '
                    'weights or manually balance by undersampling majority class and '
                    'oversampling minority class. This is applicable to all of the '
                    'learning algorithms here, with the exception of KNN which has no '
                    'concept of penalty weights. This flag will instead avoid balancing '
                    'the data if KNN is used.')
parser.add_argument('-nt', '--num_trials', dest='num_trials',
                    type=int, default=1,
                    help='Number of times to redo the test with each threshold. Each '
                    'trial will use different train and test set and the average values '
                    'across trials will be used to reduce noise')
parser.add_argument('-np', '--num_processes', dest='num_processes',
                    type = int, default = 1, 
                    help='The number of processes to start in parallel. Each process '
                    'will compute metrics from one trial. Different trials will have '
                    'different test and train split, but they will try different train '
                    'sizes.')
parser.add_argument('-sfs', '--skip_feature_selection', dest='skip_feature_selection',
                    default=False, action='store_true',
                    help='If specified, skips feature selection. Default is to run '
                    'cross validated feature selection on original features and '
                    'transform data into a smaller dimension.')
parser.add_argument('-sgs', '--skip_grid_search', dest='skip_grid_search',
                    default=False, action='store_true',
                    help='If specified, skips grid search. Default is to run cross '
                    'validation with grid search to determine the best parameters. Then '
                    'run the training once more with best params. If specified, this '
                    'step is skipped, and the provided parameters are used to train the '
                    'model. Default is False which performs the grid search.')
parser.add_argument('input_filename')
parser.add_argument('output_filename')

# The following flags will be unused if grid search is not skipped.
# SVM Flags
parser.add_argument('-sc', '--svm_cost', dest='svm_cost', type=float, default=1)
parser.add_argument('-sk', '--svm_kernel', dest='svm_kernel',
                    choices = ['rbf', 'poly', 'linear'], default='rbf')
parser.add_argument('-sg', '--svm_gamma', dest='svm_gamma', type=float,
                    default=0.1)
parser.add_argument('-sd', '--svm_degree', dest='svm_degree', type=int,
                    default=10)

# Random Forest Flags
parser.add_argument('-rnt', '--rf_num_trees', dest='rf_num_trees', type=int,
                    default=400)
parser.add_argument('-rc', '--rf_criterion', dest='rf_criterion',
                    default='gini', choices=['gini', 'entropy'])
parser.add_argument('-rmf', '--rf_max_features', dest='rf_max_features',
                    default='auto')
parser.add_argument('-rmss', '--rf_min_samples_split', dest='rf_min_samples_split',
                    default=2, type=int)
parser.add_argument('-rmsl', '--rf_min_samples_leaf', dest='rf_min_samples_leaf',
                    default=1, type=int)

# Logistic Regression Flags
parser.add_argument('-lop', '--logistic_penalty', dest='logistic_penalty',
                    default='l2', choices=['l1','l2'])
parser.add_argument('-lod', '--logistic_dual', dest='logistic_dual',
                    default=False, action='store_true',
                    help='Dual formulation is only implemented for l2 penalty')
parser.add_argument('-loc', '--logistic_cost', dest='logistic_cost',
                    default=1.0, type=float)
parser.add_argument('-lot', '--logistic_tolerance', dest='logistic_tolerance',
                    default=0.0005, type=float,
                    help='Tolerance for the stopping criteria of logistic solver.')

# KNN Flags
parser.add_argument('-knn', '--knn_num_neighbors', dest='knn_num_neighbors',
                    default=5, type=int,
                    help='The number of nearest neighbors to use with KNN.')
parser.add_argument('-kw', '--knn_weights', dest='knn_weights',
                    default='distance', choices=['uniform', 'distance'],
                    help='The KNN scheme according to which weights are given to '
                    'neighbors. Default is distance.')
parser.add_argument('-ka', '--knn_algorithm', dest='knn_algorithm',
                    default='auto', choices=['auto', 'ball_tree', 'kd_tree', 'brute'],
                    help='KNN Algorithm used to find nearest neighbors. Default is auto.')
parser.add_argument('-km', '--knn_metric', dest='knn_metric',
                    default='euclidean',
                    choices=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                    help='The distance metric to use for evaluating neighbors in KNN. '
                    'Default is euclidean.')
parser.add_argument('-kp', '--knn_power', dest='knn_power',
                    default='2', type=int,
                    help='The power parameter to use if minkowski distance is used in '
                    'KNN.')
args = parser.parse_args()

def compute_trial_metrics(df, filtering_thresholds, filtering_column, random_seed):
  random.seed(random_seed)

  # mapping from filtering threshold to any of "accuracy", "precision"... to a value
  trial_metrics = defaultdict(dict)
  for filtering_threshold in filtering_thresholds:
    # first extract the piece of data satisfying the requested threshold and then split to
    # test/train
    filtered_df = df[df[filtering_column] >= filtering_threshold]
    percentage_data = (100.0 * len(filtered_df.index)) / len(df.index)
    data = filtered_df.values
    features = data[:, 0:args.label_column]
    labels = data[:, args.label_column].astype(int)
    train_features, test_features, train_labels, test_labels = (
        model_selection.train_test_split(features, labels, test_size=args.test_size,
                                         random_state=random.randint(1,99999999)))
    assert train_features.shape[0] >= args.train_size

    if args.learning_algorithm == 'random-forest':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_random_forest(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          args.skip_feature_selection, args.skip_grid_search,
          args.rf_max_features, args.rf_num_trees, args.rf_criterion,
          args.rf_min_samples_split, args.rf_min_samples_leaf,
          args.num_processes)
    elif args.learning_algorithm == 'svm':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_svm(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          'minmax', 0, 1,
          args.skip_feature_selection, args.skip_grid_search,
          args.svm_kernel, args.svm_gamma, args.svm_cost, args.svm_degree,
          args.num_processes)
    elif args.learning_algorithm == 'logistic':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_logistic(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          args.skip_feature_selection, args.skip_grid_search,
          args.logistic_penalty, args.logistic_cost, args.logistic_dual,
          args.logistic_tolerance,
          args.num_processes)
    elif args.learning_algorithm == 'knn':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_knn(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          'minmax', 0, 1,
          args.skip_feature_selection, args.skip_grid_search,
          args.knn_num_neighbors, args.knn_weights, args.knn_algorithm,
          args.knn_metric,
          args.num_processes)
    else:
      sys.exit('bad algorithm name.')

    y_true, y_pred = test_labels, model.predict(transformed_test_features)
    predicted_probabilities = model.predict_proba(transformed_test_features)

    # size of labels in train/test.
    train_size = transformed_train_features.shape[0]
    train_female_size = sum(transformed_train_labels[:] == utils.FEMALE)
    train_male_size = sum(transformed_train_labels[:] == utils.MALE)
    test_size = transformed_test_features.shape[0]
    test_female_size = sum(test_labels[:] == utils.FEMALE)
    test_male_size = sum(test_labels[:] == utils.MALE)
    
    # Compute evaluation metrics
    test_accuracy = model.score(transformed_test_features, test_labels)*100.
    test_AUC = roc_auc_score(y_true, predicted_probabilities[:,1])
    (precisions, recalls, f1scores, supports) = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=[0, 1])
    (ave_precision, ave_recall, ave_f1score, ave_support) = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=[0, 1], average='macro')
    # Get true/false positive/negative
    confusion = confusion_matrix(y_true, y_pred)
    test_true_female = confusion[utils.FEMALE][utils.FEMALE]
    test_false_female = confusion[utils.MALE][utils.FEMALE]
    test_true_male = confusion[utils.MALE][utils.MALE]
    test_false_male = confusion[utils.FEMALE][utils.MALE]

    trial_metrics[filtering_threshold]["min_active_days"] = filtering_threshold
    trial_metrics[filtering_threshold]["percentage_data"] = percentage_data
    trial_metrics[filtering_threshold]["train_size"] = train_size
    trial_metrics[filtering_threshold]["train_female_size"] = train_female_size
    trial_metrics[filtering_threshold]["train_male_size"] = train_male_size
    trial_metrics[filtering_threshold]["test_size"] = test_size
    trial_metrics[filtering_threshold]["test_female_size"] = test_female_size
    trial_metrics[filtering_threshold]["test_male_size"] = test_male_size
    trial_metrics[filtering_threshold]["test_true_female"] = test_true_female
    trial_metrics[filtering_threshold]["test_false_female"] = test_false_female
    trial_metrics[filtering_threshold]["test_true_male"] = test_true_male
    trial_metrics[filtering_threshold]["test_false_male"] = test_false_male
    trial_metrics[filtering_threshold]["test_accuracy"] = test_accuracy
    trial_metrics[filtering_threshold]["test_AUC"] = test_AUC
    trial_metrics[filtering_threshold]["test_average_precision"] = ave_precision
    trial_metrics[filtering_threshold]["test_female_precision"] = precisions[utils.FEMALE]
    trial_metrics[filtering_threshold]["test_male_precision"] = precisions[utils.MALE]
    trial_metrics[filtering_threshold]["test_average_recall"] = ave_recall
    trial_metrics[filtering_threshold]["test_female_recall"] = recalls[utils.FEMALE]
    trial_metrics[filtering_threshold]["test_male_recall"] = recalls[utils.MALE]
    trial_metrics[filtering_threshold]["test_average_f1score"] = ave_f1score
    trial_metrics[filtering_threshold]["test_female_f1score"] = f1scores[utils.FEMALE]
    trial_metrics[filtering_threshold]["test_male_f1score"] = f1scores[utils.MALE]
 
  return trial_metrics


def main():
  df = pandas.read_csv(args.input_filename, index_col=False, header=0)
  
  # figure out the name of the filtering column
  column_names = df.columns.values.tolist()
  filtering_columns = ['number_of_contacts__allweek__allday__call__mean',
                       'number_of_contacts__call__mean']
  filtering_columns = [x for x in filtering_columns if x in column_names]
  if len(filtering_columns) == 2:
    sys.exit('Both columns ' + str(filtering_columns) + ' are present in data.')
  if len(filtering_columns) == 0:
    sys.exit('None of columns ' + str(filtering_columns) + ' are present in data.')
  filtering_column = filtering_columns[0]
 
  # figure out filtering thresholds that satisfy the requested train and test sizes. Need
  # to do a dummy split to train and test, so we can figure out max possible train size
  # after balancing
  filtering_thresholds = list()
  for filtering_threshold in arange(0,7,0.5):
    data = df[df[filtering_column] >= filtering_threshold].values
    features = data[:, 0:args.label_column]
    labels = data[:, args.label_column].astype(int)
    dummy_train_features, dummy_test_features, dummy_train_labels, dummy_test_labels = (
      model_selection.train_test_split(features, labels, test_size=args.test_size))
    dummy_train_features, dummy_train_labels, penalty_weights = utils.prepare_train_data(
        dummy_train_features, dummy_train_labels, args.scikit_balancing, -1)
    # This is a good filtering threshold if the number of data points satisfying the
    # threshold exceeds the requested train size
    if dummy_train_features.shape[0] >= args.train_size:
      filtering_thresholds.append(filtering_threshold)
    else:
      break
  
  metric_names = ["min_active_days", "percentage_data",
                  "train_size", "train_female_size", "train_male_size",
                  "test_size", "test_female_size", "test_male_size",
                  "test_true_female", "test_false_female",
                  "test_true_male", "test_false_male",
                  "test_accuracy", "test_AUC",
                  "test_average_precision",
                  "test_female_precision", "test_male_precision",
                  "test_average_recall",
                  "test_female_recall", "test_male_recall",
                  "test_average_f1score",
                  "test_female_f1score", "test_male_f1score"]
  # mapping from filtering threshold to any of "accuracy", "precision"... to list of
  # values, each value corresponding to the result from one trial
  results = defaultdict(lambda: defaultdict(list))
  for trial in range(args.num_trials):
    random_seed = random.randint(1, 999999999)
    trial_metrics = compute_trial_metrics(df, filtering_thresholds, filtering_column,
                                          random_seed)

    # loop over different filtering thresholds in dict
    for filtering_threshold in filtering_thresholds:
      metric_values = trial_metrics[filtering_threshold]
      # loop over different metrics
      for metric in metric_names:
        results[filtering_threshold][metric].append(metric_values[metric])
          
    print("\nFinished %d trials\n" % (trial+1))

  
  # generate output file and header
  output_file = open(args.output_filename, "w")
  output_file_writer = csv.writer(output_file)
  output_file_writer.writerow(metric_names)

  for filtering_threshold in filtering_thresholds:
    output_file_writer.writerow([filtering_threshold,
                                 mean(results[filtering_threshold]["percentage_data"]),
                                 int(mean(results[filtering_threshold]["train_size"])),
                                 int(mean(results[filtering_threshold]["train_female_size"])),
                                 int(mean(results[filtering_threshold]["train_male_size"])),
                                 int(mean(results[filtering_threshold]["test_size"])),
                                 int(mean(results[filtering_threshold]["test_female_size"])),
                                 int(mean(results[filtering_threshold]["test_male_size"])),
                                 int(mean(results[filtering_threshold]["test_true_female"])),
                                 int(mean(results[filtering_threshold]["test_false_female"])),
                                 int(mean(results[filtering_threshold]["test_true_male"])),
                                 int(mean(results[filtering_threshold]["test_false_male"])),
                                 mean(results[filtering_threshold]["test_accuracy"]),
                                 mean(results[filtering_threshold]["test_AUC"]),
                                 mean(results[filtering_threshold]["test_average_precision"]),
                                 mean(results[filtering_threshold]["test_female_precision"]),
                                 mean(results[filtering_threshold]["test_male_precision"]),
                                 mean(results[filtering_threshold]["test_average_recall"]),
                                 mean(results[filtering_threshold]["test_female_recall"]),
                                 mean(results[filtering_threshold]["test_male_recall"]),
                                 mean(results[filtering_threshold]["test_average_f1score"]),
                                 mean(results[filtering_threshold]["test_female_f1score"]),
                                 mean(results[filtering_threshold]["test_male_f1score"])
                                ])
  output_file.close() 

if __name__ == "__main__":
  main()
