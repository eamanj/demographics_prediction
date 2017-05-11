#!/usr/bin/python
 argparse
import pandas
import utils
import csv
import math
import random
import sys
import multiprocessing
import models
from collections import defaultdict
from numpy import mean, random

from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

parser = argparse.ArgumentParser(
    description='This script test prediction on a fixed test size at different training '
                'size. The idea is how small of training one requires to reach the '
                'bottom-line accuracy. The less the better')
parser.add_argument('-ts', '--test_size', dest='test_size',
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
                    help='Number of times to redo the test with each train size. Each '
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

def compute_trial_metrics(data):
  features = data[0]
  labels = data[1]
  train_sizes = data[2]
  random_seed = data[3]

  random.seed(random_seed)
  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=args.test_size,
                                        random_state=random.randint(1,99999999)))

  # mapping from train size to any of "accuracy", "precision"... to a value
  trial_metrics = defaultdict(dict)
  for train_size in train_sizes:
    if args.learning_algorithm == 'random-forest':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_random_forest(
          train_features, train_labels, test_features,
          args.scikit_balancing, train_size,
          args.skip_feature_selection, args.skip_grid_search,
          args.rf_max_features, args.rf_num_trees, args.rf_criterion,
          args.rf_min_samples_split, args.rf_min_samples_leaf, 1)
    elif args.learning_algorithm == 'svm':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_svm(
          train_features, train_labels, test_features,
          args.scikit_balancing, train_size,
          'minmax', 0, 1,
          args.skip_feature_selection, args.skip_grid_search,
          args.svm_kernel, args.svm_gamma, args.svm_cost, args.svm_degree, 1)
    elif args.learning_algorithm == 'logistic':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_logistic(
          train_features, train_labels, test_features,
          args.scikit_balancing, train_size,
          args.skip_feature_selection, args.skip_grid_search,
          args.logistic_penalty, args.logistic_cost, args.logistic_dual,
          args.logistic_tolerance, 1)
    elif args.learning_algorithm == 'knn':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_knn(
          train_features, train_labels, test_features,
          args.scikit_balancing, train_size,
          'minmax', 0, 1,
          args.skip_feature_selection, args.skip_grid_search,
          args.knn_num_neighbors, args.knn_weights, args.knn_algorithm,
          args.knn_metric, 1)
    else:
      sys.exit('bad algorithm name.')

    y_true, y_pred = test_labels, model.predict(transformed_test_features)

    # size of labels in train/test.
    test_size = transformed_test_features.shape[0]
    test_female_size = sum(test_labels[:] == utils.FEMALE)
    test_male_size = sum(test_labels[:] == utils.MALE)
    
    # Compute evaluation metrics
    test_accuracy = model.score(transformed_test_features, test_labels)*100.
    (precisions, recalls, fscores, supports) = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=[0, 1])
    # Get true/false positive/negative
    confusion = confusion_matrix(y_true, y_pred)
    test_true_female = confusion[utils.FEMALE][utils.FEMALE]
    test_false_female = confusion[utils.MALE][utils.FEMALE]
    test_true_male = confusion[utils.MALE][utils.MALE]
    test_false_male = confusion[utils.FEMALE][utils.MALE]

    trial_metrics[train_size]["train_size"] = train_size
    trial_metrics[train_size]["test_size"] = test_size
    trial_metrics[train_size]["test_female_size"] = test_female_size
    trial_metrics[train_size]["test_male_size"] = test_male_size
    trial_metrics[train_size]["test_true_female"] = test_true_female
    trial_metrics[train_size]["test_false_female"] = test_false_female
    trial_metrics[train_size]["test_true_male"] = test_true_male
    trial_metrics[train_size]["test_false_male"] = test_false_male
    trial_metrics[train_size]["test_accuracy"] = test_accuracy
    trial_metrics[train_size]["test_female_precision"] = precisions[utils.FEMALE]
    trial_metrics[train_size]["test_male_precision"] = precisions[utils.MALE]
    trial_metrics[train_size]["test_female_recall"] = recalls[utils.FEMALE]
    trial_metrics[train_size]["test_male_recall"] = recalls[utils.MALE]
  
  return trial_metrics


def main():
  df = pandas.read_csv(args.input_filename, index_col=False, header=0)
  data = df.values
  column_names = df.columns.values.tolist()
  
  # Extract features/labels and their names from raw data
  features = data[:, 0:args.label_column]
  labels = data[:, args.label_column].astype(int)
  feature_names = column_names[0:args.label_column]
  label_name = column_names[args.label_column]
  
  # We specify absolute train sizes so we can compare across different data sets with
  # different overal sizes. Need to do a dummy split to train and test, so we can figure
  # out max possible train size after balancing
  dummy_train_features, dummy_test_features, dummy_train_labels, dummy_test_labels = (
    model_selection.train_test_split(features, labels, test_size=args.test_size))
  dummy_train_features, dummy_train_labels, penalty_weights = utils.prepare_train_data(
      dummy_train_features, dummy_train_labels, args.scikit_balancing, -1)
  max_possible_train_size = dummy_train_features.shape[0]
  train_sizes = range(400, 15000, 100)
  train_sizes.extend(range(15000, min(max_possible_train_size, 30001), 500))
 
  metric_names = ["train_size",
                  "test_size", "test_female_size", "test_male_size",
                  "test_true_female", "test_false_female",
                  "test_true_male", "test_false_male",
                  "test_accuracy",
                  "test_female_precision", "test_male_precision",
                  "test_female_recall", "test_male_recall"]
  # mapping from train size to any of "accuracy", "precision"... to list of values, each
  # value corresponding to the result from one trial
  results = defaultdict(lambda: defaultdict(list))
  finished_trials = 0
  while finished_trials < args.num_trials:
    # Figure out how many parallel processes we should launch to satisfy number of trials.
    num_processes = min(args.num_processes, args.num_trials - finished_trials)
    replicated_data = list()
    for n in range(0, num_processes):
      # VERY IMPORTANT: Provide a random state, since it seems like multiple workers split
      # the data in the same way due to an identical intitial random state
      random_seed = random.randint(1, 999999999)
      replicated_data.append((features, labels, train_sizes, random_seed))

    pool = multiprocessing.Pool(processes = num_processes)
    trials_metrics = pool.map(compute_trial_metrics, replicated_data)
    pool.close()
    finished_trials += num_processes

    # Add trial metrics to results by looping over different trials in a list
    for trial_metrics in trials_metrics:
      # loop over different train size in dict
      for train_size in train_sizes:
        metric_values = trial_metrics[train_size]
        # loop over different metrics
        for metric in metric_names:
          results[train_size][metric].append(metric_values[metric])
          
    print("\nFinished %d trials\n" % finished_trials)

  
  # generate output file and header
  output_file = open(args.output_filename, "w")
  output_file_writer = csv.writer(output_file)
  output_file_writer.writerow(metric_names)

  for train_size in train_sizes:
    output_file_writer.writerow([train_size,
                                 int(mean(results[train_size]["test_size"])),
                                 int(mean(results[train_size]["test_female_size"])),
                                 int(mean(results[train_size]["test_male_size"])),
                                 int(mean(results[train_size]["test_true_female"])),
                                 int(mean(results[train_size]["test_false_female"])),
                                 int(mean(results[train_size]["test_true_male"])),
                                 int(mean(results[train_size]["test_false_male"])),
                                 mean(results[train_size]["test_accuracy"]),
                                 mean(results[train_size]["test_female_precision"]),
                                 mean(results[train_size]["test_male_precision"]),
                                 mean(results[train_size]["test_female_recall"]),
                                 mean(results[train_size]["test_male_recall"])
                                ])
  output_file.close() 

if __name__ == "__main__":
  main()
