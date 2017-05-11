#!/usr/bin/python

import argparse
import pandas
import utils
import csv
import math
import random
import sys
import numpy
import multiprocessing
import models
from collections import defaultdict
from numpy import mean, random

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

parser = argparse.ArgumentParser(
    description='This script uses the train data in its original distribution, and tests '
                'it on a test data with varying ratio of males to females. The goal is '
                'to evaluate the performance of algorithm in retreive the correction '
                'proportion of males to females in any test data.')
parser.add_argument('-trs', '--train_size', dest='train_size',
                    type=int, default=-1,
                    help='The size of train set. It must be a whole integer, greater '
                    'than 1 which represents the absolute number of data points to use '
                    'for training. Default is -1 which means all available data after '
                    'splitting into test and train is used. There will be some '
                    'undersampling of the majority class, if train data needs to be '
                    'balanced.')
parser.add_argument('-ts', '--test_size', dest='test_size',
                    type=int, default = 15000,
                    help='The number of data points to keep aside as test. The number '
                    'of males to females will change according to the ratio, but their '
                    'total count will always be this number. '
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
parser.add_argument('-rnj', '--rf_num_jobs', dest='rf_num_jobs', type=int, default=-1,
                    help='Number of jobs for building trees and testing in parallel. '
                    'Default is -1, which corresponds to the number of cores in the '
                    'machine')
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
  female_features = data[0]
  female_labels = data[1]
  male_features = data[2]
  male_labels = data[3]
  test_actual_ratios = data[4]
  random_seed = data[5]

  random.seed(random_seed)
  
  num_males = male_features.shape[0]
  num_females = female_features.shape[0]

  # Make sure you seed the random state since each subprocess will receive the same state,
  # so all random numbers will become identical!
  numpy.random.seed(random_seed)
  # mapping from train size to any of "accuracy", "precision"... to a value
  trial_metrics = defaultdict(dict)
  for test_actual_ratio in test_actual_ratios:
    # construct test set with given ratio of female to test size
    test_female_size = int(1.0*args.test_size*test_actual_ratio)
    test_male_size = args.test_size - test_female_size 
  
    if num_females < test_female_size:
      sys.exit('Not enough female samples: ' + str(test_female_size) + ' for ratio: ' +
               str(test_actual_ratio))
    if num_males < test_male_size:
      sys.exit('Not enough male samples: ' + str(test_male_size) + ' for ratio: ' +
               str(test_actual_ratio))
    
    test_female_indices = numpy.random.choice(num_females, test_female_size,
                                              replace=False)
    test_male_indices = numpy.random.choice(num_males, test_male_size,
                                            replace=False)
    female_mask = numpy.zeros(num_females, dtype=bool)
    female_mask[test_female_indices] = True
    male_mask = numpy.zeros(num_males, dtype=bool)
    male_mask[test_male_indices] = True
    
    test_female_features = female_features[female_mask, :]
    test_female_labels = female_labels[female_mask]
    test_male_features = male_features[male_mask, :]
    test_male_labels = male_labels[male_mask]
   
    train_female_features = female_features[~female_mask, :]
    train_female_labels = female_labels[~female_mask]
    train_male_features = male_features[~male_mask, :]
    train_male_labels = male_labels[~male_mask]

    test_features = numpy.concatenate((test_female_features, test_male_features))
    test_labels = numpy.concatenate((test_female_labels, test_male_labels))
    
    train_features = numpy.concatenate((train_female_features, train_male_features))
    train_labels = numpy.concatenate((train_female_labels, train_male_labels))

    if args.learning_algorithm == 'random-forest':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_random_forest(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          args.skip_feature_selection, args.skip_grid_search,
          args.rf_max_features, args.rf_num_trees, args.rf_criterion,
          args.rf_min_samples_split, args.rf_min_samples_leaf, 1)
    elif args.learning_algorithm == 'svm':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_svm(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          'minmax', 0, 1,
          args.skip_feature_selection, args.skip_grid_search,
          args.svm_kernel, args.svm_gamma, args.svm_cost, args.svm_degree, 1)
    elif args.learning_algorithm == 'logistic':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_logistic(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          args.skip_feature_selection, args.skip_grid_search,
          args.logistic_penalty, args.logistic_cost, args.logistic_dual,
          args.logistic_tolerance, 1)
    elif args.learning_algorithm == 'knn':
      (model, transformed_train_features, transformed_train_labels, transformed_test_features) = models.train_knn(
          train_features, train_labels, test_features,
          args.scikit_balancing, args.train_size,
          'minmax', 0, 1,
          args.skip_feature_selection, args.skip_grid_search,
          args.knn_num_neighbors, args.knn_weights, args.knn_algorithm,
          args.knn_metric, 1)
    else:
      sys.exit('bad algorithm name.')


    # size of labels in train/test
    train_size = transformed_train_features.shape[0]
    train_female_size = sum(transformed_train_labels[:] == utils.FEMALE)
    train_male_size = sum(transformed_train_labels[:] == utils.MALE)
    test_size = transformed_test_features.shape[0]
    test_actual_female_size = sum(test_labels[:] == utils.FEMALE)
    test_actual_male_size = sum(test_labels[:] == utils.MALE)
   
    # train performance
    y_true, y_pred = transformed_train_labels, model.predict(transformed_train_features)
    train_predicted_female_size = sum(y_pred[:] == utils.FEMALE)
    train_predicted_male_size = sum(y_pred[:] == utils.MALE)
    confusion = confusion_matrix(y_true, y_pred)
    train_true_female = confusion[utils.FEMALE][utils.FEMALE]
    train_false_female = confusion[utils.MALE][utils.FEMALE]
    train_true_male = confusion[utils.MALE][utils.MALE]
    train_false_male = confusion[utils.FEMALE][utils.MALE]
    train_accuracy = model.score(transformed_train_features, transformed_train_labels)*100.
    (train_precisions, train_recalls, fscores, supports) = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=[0, 1])

    # test performance
    y_true, y_pred = test_labels, model.predict(transformed_test_features)
    test_predicted_female_size = sum(y_pred[:] == utils.FEMALE)
    test_predicted_male_size = sum(y_pred[:] == utils.MALE)
    test_predicted_ratio = (1.0*test_predicted_female_size)/test_size
    confusion = confusion_matrix(y_true, y_pred)
    test_true_female = confusion[utils.FEMALE][utils.FEMALE]
    test_false_female = confusion[utils.MALE][utils.FEMALE]
    test_true_male = confusion[utils.MALE][utils.MALE]
    test_false_male = confusion[utils.FEMALE][utils.MALE]
    test_accuracy = model.score(transformed_test_features, test_labels)*100.
    (test_precisions, test_recalls, fscores, supports) = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=[0, 1])

    trial_metrics[test_actual_ratio]["train_size"] = train_size
    trial_metrics[test_actual_ratio]["train_female_size"] = train_female_size
    trial_metrics[test_actual_ratio]["train_male_size"] = train_male_size
    trial_metrics[test_actual_ratio]["train_predicted_female_size"] = train_predicted_female_size
    trial_metrics[test_actual_ratio]["train_predicted_male_size"] = train_predicted_male_size
    trial_metrics[test_actual_ratio]["train_true_female"] = train_true_female
    trial_metrics[test_actual_ratio]["train_false_female"] = train_false_female
    trial_metrics[test_actual_ratio]["train_true_male"] = train_true_male
    trial_metrics[test_actual_ratio]["train_false_male"] = train_false_male
    trial_metrics[test_actual_ratio]["train_accuracy"] = train_accuracy
    trial_metrics[test_actual_ratio]["train_female_precision"] = train_precisions[utils.FEMALE]
    trial_metrics[test_actual_ratio]["train_male_precision"] = train_precisions[utils.MALE]
    trial_metrics[test_actual_ratio]["train_female_recall"] = train_recalls[utils.FEMALE]
    trial_metrics[test_actual_ratio]["train_male_recall"] = train_recalls[utils.MALE]
    trial_metrics[test_actual_ratio]["test_size"] = test_size
    trial_metrics[test_actual_ratio]["test_actual_ratio"] = test_actual_ratio
    trial_metrics[test_actual_ratio]["test_actual_female_size"] = test_actual_female_size
    trial_metrics[test_actual_ratio]["test_actual_male_size"] = test_actual_male_size
    trial_metrics[test_actual_ratio]["test_predicted_ratio"] = test_predicted_ratio
    trial_metrics[test_actual_ratio]["test_predicted_female_size"] = test_predicted_female_size
    trial_metrics[test_actual_ratio]["test_predicted_male_size"] = test_predicted_male_size
    trial_metrics[test_actual_ratio]["test_true_female"] = test_true_female
    trial_metrics[test_actual_ratio]["test_false_female"] = test_false_female
    trial_metrics[test_actual_ratio]["test_true_male"] = test_true_male
    trial_metrics[test_actual_ratio]["test_false_male"] = test_false_male
    trial_metrics[test_actual_ratio]["test_accuracy"] = test_accuracy
    trial_metrics[test_actual_ratio]["test_female_precision"] = test_precisions[utils.FEMALE]
    trial_metrics[test_actual_ratio]["test_male_precision"] = test_precisions[utils.MALE]
    trial_metrics[test_actual_ratio]["test_female_recall"] = test_recalls[utils.FEMALE]
    trial_metrics[test_actual_ratio]["test_male_recall"] = test_recalls[utils.MALE]

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
 
  # separate males and females
  male_indices = (labels[:] == utils.MALE)
  female_indices = (labels[:] == utils.FEMALE)
  male_features = features[male_indices]
  female_features = features[female_indices]
  male_labels = labels[male_indices]
  female_labels = labels[female_indices]

  # list of female to test size ratios we will test for
  test_actual_ratios = numpy.arange(0, 1.001, 0.1)

  metric_names = ["train_size", "train_female_size", "train_male_size",
                  "train_predicted_female_size", "train_predicted_male_size",
                  "train_true_female", "train_false_female",
                  "train_true_male", "train_false_male",
                  "train_accuracy",
                  "train_female_precision", "train_male_precision",
                  "train_female_recall", "train_male_recall",
                  "test_size",
                  "test_actual_ratio", "test_actual_female_size", "test_actual_male_size",
                  "test_predicted_ratio",
                  "test_predicted_female_size", "test_predicted_male_size",
                  "test_true_female", "test_false_female",
                  "test_true_male", "test_false_male",
                  "test_accuracy",
                  "test_female_precision", "test_male_precision",
                  "test_female_recall", "test_male_recall"]
  # mapping from test_actual_ratio to any of "accuracy", "precision"... to list of
  # values, each value corresponding to the result from one trial
  results = defaultdict(lambda: defaultdict(list))
  finished_trials = 0
  while finished_trials < args.num_trials:
    # Figure out how many parallel processes we should launch to satisfy number of trials.
    num_processes = min(args.num_processes, args.num_trials - finished_trials)
    # Replicate the data for processes
    replicated_data = list()
    for n in range(0, num_processes):
      # VERY IMPORTANT: Provide a random state, since it seems like multiple workers split
      # the data in the same way due to an identical intitial random state
      random_seed = random.randint(1, 999999999)
      replicated_data.append((female_features, female_labels,
                              male_features, male_labels,
                              test_actual_ratios, random_seed))
    pool = multiprocessing.Pool(processes = num_processes)
    trials_metrics = pool.map(compute_trial_metrics, replicated_data)
    pool.close()
    finished_trials += num_processes

    # Add trial metrics to results by looping over different trials in a list
    for trial_metrics in trials_metrics:
      # loop over different test ratio in dict
      for test_actual_ratio in test_actual_ratios:
        metric_values = trial_metrics[test_actual_ratio]
        # loop over different metrics
        for metric in metric_names:
          results[test_actual_ratio][metric].append(metric_values[metric])
          
    print("\nFinished %d trials\n" % finished_trials)

  
  # generate output file and header
  output_file = open(args.output_filename, "w")
  output_file_writer = csv.writer(output_file)
  output_file_writer.writerow(metric_names)

  for test_actual_ratio in test_actual_ratios:
    output_file_writer.writerow(
        [int(mean(results[test_actual_ratio]["train_size"])),
         int(mean(results[test_actual_ratio]["train_female_size"])),
         int(mean(results[test_actual_ratio]["train_male_size"])),
         int(mean(results[test_actual_ratio]["train_predicted_female_size"])),
         int(mean(results[test_actual_ratio]["train_predicted_male_size"])),
         int(mean(results[test_actual_ratio]["train_true_female"])),
         int(mean(results[test_actual_ratio]["train_false_female"])),
         int(mean(results[test_actual_ratio]["train_true_male"])),
         int(mean(results[test_actual_ratio]["train_false_male"])),
         mean(results[test_actual_ratio]["train_accuracy"]),
         mean(results[test_actual_ratio]["train_female_precision"]),
         mean(results[test_actual_ratio]["train_male_precision"]),
         mean(results[test_actual_ratio]["train_female_recall"]),
         mean(results[test_actual_ratio]["train_male_recall"]),
         int(mean(results[test_actual_ratio]["test_size"])),
         test_actual_ratio,
         int(mean(results[test_actual_ratio]["test_actual_female_size"])),
         int(mean(results[test_actual_ratio]["test_actual_male_size"])),
         mean(results[test_actual_ratio]["test_predicted_ratio"]),
         int(mean(results[test_actual_ratio]["test_predicted_female_size"])),
         int(mean(results[test_actual_ratio]["test_predicted_male_size"])),
         int(mean(results[test_actual_ratio]["test_true_female"])),
         int(mean(results[test_actual_ratio]["test_false_female"])),
         int(mean(results[test_actual_ratio]["test_true_male"])),
         int(mean(results[test_actual_ratio]["test_false_male"])),
         mean(results[test_actual_ratio]["test_accuracy"]),
         mean(results[test_actual_ratio]["test_female_precision"]),
         mean(results[test_actual_ratio]["test_male_precision"]),
         mean(results[test_actual_ratio]["test_female_recall"]),
         mean(results[test_actual_ratio]["test_male_recall"])
        ])
  output_file.close() 

if __name__ == "__main__":
  main()
