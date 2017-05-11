#!/usr/bin/python

import argparse
import csv
import pandas
import numpy
import sys
import utils
import feature_selection
import warnings
import random
import multiprocessing

from sklearn import model_selection, svm
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

parser = argparse.ArgumentParser(
    description='This script selects top K features using linear SVM with L1 '
                'penalty then performs the learning task on the set of top scoring '
                'features and reports the test accuracy. There will two learning tasks '
                'involved. The first learner is used for extracting important features. '
                'The second learner measures the test accuracy on the selected '
                'features by the first learner. Under the hood, the number of selected '
                'features selected is determined by the cost/regularization parameter '
                'for the feature selector SVM which is an L1-svm. But you don\'t specify '
                'the cost paramter that yields a specific number of selected features. '
                'Instead, you can specify a range for the number of features to extract. '
                'Algorithm will run multiple times, extracts top k features for any k '
                'between min_num_features and max_num_features and reports the results.')
parser.add_argument('input_filename')
parser.add_argument('output_filename')
parser.add_argument('-min_nf', '--min_num_features', dest='min_num_features',
                    type=int, default=1,
                    help='Minimum Number of features to select in first phase by L1-SVM.')
parser.add_argument('-max_nf', '--max_num_features', dest='max_num_features',
                    type=int, default=94,
                    help='Maximum Number of features to select in first phase by L1-SVM.')
parser.add_argument('-ns', '--num_samples', dest='num_samples', type=int,
                    default=20000,
                    help='Number of samples to use in trials of feature selection tasks. '
                    'There will be many trials of L1-svm to search for the right '
                    'regularization parameter that yields the requested number of '
                    'features. So we do that not the whole sample, but on a smaller one '
                    'to speed up the whole process.')
parser.add_argument('-ts', '--test_size', dest='test_size',
                    type=float, default=0.2,
                    help='The size of test set. If a float between 0 and 1, it '
                    'represents the fraction of input data points to use for test. '
                    'If an integer greater than 1, it represents the absolute number of '
                    'data points to use for test.')
parser.add_argument('-la', '--learning_algorithm', dest='learning_algorithm',
                    choices=['svm', 'random-forest', 'logistic', 'knn', 'none'],
                    default='svm',
                    help = 'Determines the learning algorithm to employ after omitting '
                    'the low scoring features. Possible values are svm random forest '
                    'and logistic regression. Default is svm.')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=94,
                    help='The column number of the label in the input csv. Default is '
                    '94, set it otherwise')
parser.add_argument('-sb', '--scikit_balancing', dest='scikit_balancing',
                    default=False, action='store_true',
                    help='Whether to use scikit data balancing by changing sample '
                    'weights or manually balance by undersampling majority class and '
                    'oversampling minority class')

# SVM Flags
parser.add_argument('-c', '--cost', dest='svm_cost', type=float, default=1,
                    help = 'The cost parameter (C) in SVM. Ignored if learning_algorithm '
                    'is random forest')
parser.add_argument('-k', '--kernel', dest='svm_kernel',
                    choices = ['rbf', 'poly', 'linear'], default='rbf',
                    help='The kernel to use in SVM. Choices are rbf, poly and linear. '
                    'Ignored if learning_algorithm is random forest')
parser.add_argument('-g', '--gamma', dest='svm_gamma', type=float,
                    default=0.1,
                    help='The kernel coefficient for poly and rbf kernels. default is '
                    '0.1. Ignored if learning_algorithm is random forest')
parser.add_argument('-d', '--degree', dest='svm_degree', type=int,
                    default=3,
                    help='The degree of polynomial kernel. default is 3. Ignored if '
                    'learning_algorithm is random forest.')

# Random Forest Flags
parser.add_argument('-nt', '--num_trees', dest='rf_num_trees', type=int,
                    default=10,
                    help = 'The number of trees in the forest. Ignored if '
                    'learning_algorithm is svm.')
parser.add_argument('-cr', '--criterion', dest='rf_criterion',
                    default='gini', choices=['gini', 'entropy'],
                    help='The criterion used for evaluating the trees. Possible values '
                    'or "gini" and "entropy". Ignored if learning_algorithm is svm.')
parser.add_argument('-mf', '--max_features', dest='rf_max_features',
                    default='auto',
                    help='The number of features to consider when looking for the best '
                    'split. Possible values are: "auto", "sqrt", "log2", "all", where '
                    'all uses all features. Ignored if learning_algorithm is svm.')
parser.add_argument('-mss', '--min_samples_split', dest='rf_min_samples_split',
                    default=2, type=int,
                    help='The minimum number of samples required to split an internal '
                    'node. Must be a non-negative integer greater than min_samples_leaf. '
                    'Ignored if learning_algorithm is svm.')
parser.add_argument('-msl', '--min_samples_leaf', dest='rf_min_samples_leaf',
                    default=1, type=int,
                    help='The minimum number of samples that must exist in a leaf for a '
                    'split to occur. Must be a non-negative integer. Ignored if '
                    'learning_algorithm is svm.')

# Logistic Regression flags
parser.add_argument('-lp', '--logistic_penalty', dest='logistic_penalty',
                    default='l2', choices=['l1', 'l2'],
                    help='The type  of norm functtion to use for the penalty in logistic '
                    'regression. Default is l2.')
parser.add_argument('-lco', '--logistic_cost', dest='logistic_cost',
                    default=1.0, type=float,
                    help='The regularization parameter to use in logistic regression. '
                    'must be a positive float. Like in support vector machines, smaller '
                    'values mean stronger regularization.')

# KNN flags
parser.add_argument('-nn', '--num_neighbors', dest='knn_num_neighbors',
                    default=5, type=int,
                    help='The number of nearest neighbors to use with KNN.')
parser.add_argument('-w', '--weights', dest='knn_weights',
                    default='distance', choices=['uniform', 'distance'],
                    help='The scheme according to which weights are given to neighbors. '
                    'in KNN. Default is distance.')
parser.add_argument('-a', '--algorithm', dest='knn_algorithm',
                    default='auto', choices=['auto', 'ball_tree', 'kd_tree', 'brute'],
                    help='Algorithm used to find nearest neighbors in KNN. Default is '
                    'auto.')
parser.add_argument('-m', '--metric', dest='knn_metric',
                    choices=['euclidean', 'manhattan', 'chebyshev'], default='euclidean',
                    help='The distance metric to use for evaluating neighbors in KNN. '
                    'Default is euclidean.')
parser.add_argument('-i', '--imbalanced_data', dest='knn_imbalanced_data',
                    default=False, action='store_true',
                    help='Whether to balance training data or not by including equal '
                    'number of positive and negative labels in training set. Default is '
                    'to balance the data.')
args = parser.parse_args()


def compute_evaluation_metrics(predicted_labels, test_labels, label_values):
  """
  Computes various evaluation metrics when predictions are binary values.

  Returns a tuple containing accuracy, and three versions of f1, precision and recall
  scores. The three versions are per-positive, per-negative and their average.
  
  predicted_labels: the labels predicted by the model.
  test_labels: the actual labels per each data point. The order of labels in this
  list must correspond to the order of predictions in test_labels.
  label_values: contains the value of each label.
  """

  positive = label_values[0]
  negative = label_values[1]
  confusion_matrix = metrics.confusion_matrix(test_labels, predicted_labels)
  true_positive = confusion_matrix[positive][positive]
  false_positive = confusion_matrix[negative][positive]
  true_negative = confusion_matrix[negative][negative]
  false_negative = confusion_matrix[positive][negative]
  test_size = true_positive + false_positive + true_negative + false_negative

  accuracy = metrics.accuracy_score(test_labels, predicted_labels)*100.
  pos_f1 = metrics.f1_score(test_labels, predicted_labels, pos_label=positive,
                            average='binary')
  neg_f1 = metrics.f1_score(test_labels, predicted_labels, pos_label=negative,
                            average='binary')
  average_f1 = metrics.f1_score(test_labels, predicted_labels, pos_label=None,
                                average='macro')

  pos_precision = metrics.precision_score(test_labels, predicted_labels,
                                          pos_label=positive, average='binary')
  neg_precision = metrics.precision_score(test_labels, predicted_labels,
                                          pos_label=negative, average='binary')
  average_precision = metrics.precision_score(test_labels, predicted_labels,
                                              pos_label=None, average='macro')

  pos_recall = metrics.recall_score(test_labels, predicted_labels, pos_label=positive,
                                    average='binary')
  neg_recall = metrics.recall_score(test_labels, predicted_labels, pos_label=negative,
                                    average='binary')
  average_recall = metrics.recall_score(test_labels, predicted_labels, pos_label=None,
                                        average='macro')

  return (test_size,
          true_positive, false_positive, true_negative, false_negative,
          accuracy,
          pos_f1, neg_f1, average_f1,
          pos_precision, neg_precision, average_precision,
          pos_recall, neg_recall, average_recall)


def perform_single_random_forest(input_data):
  """ Perform a single trial of random forest with selected features
  """
  # extract inputs from input tuple
  features = input_data[0]
  labels = input_data[1]
  rf_num_trees = input_data[2]
  rf_criterion = input_data[3]
  rf_max_features = input_data[4]
  rf_min_samples_split = input_data[5]
  rf_min_samples_leaf = input_data[6]
  scikit_balancing = input_data[7]
  test_size = input_data[8]

  # VERY IMPORTANT: Provide a random state, since it seems like multiple workers split the
  # data in the same way
  random.seed()
  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=test_size,
                                        random_state=random.randint(1,99999999)))

  train_features, train_labels, penalty_weights = utils.prepare_train_data(
      train_features, train_labels, scikit_balancing, -1)

  model = RandomForestClassifier(n_estimators=rf_num_trees,
                                 n_jobs=-1,
                                 criterion=rf_criterion,
                                 max_features=rf_max_features,
                                 min_samples_split=rf_min_samples_split,
                                 min_samples_leaf=rf_min_samples_leaf)
  model = model.fit(train_features, train_labels, sample_weight=penalty_weights)
  predicted_labels = model.predict(test_features)
  label_values = [0, 1]
  trial_metrics = compute_evaluation_metrics(predicted_labels, test_labels,
                                             label_values)
  return trial_metrics


def perform_random_forest(features,
                          labels,
                          rf_num_trees,
                          rf_criterion,
                          rf_max_features,
                          rf_min_samples_split,
                          rf_min_samples_leaf,
                          scikit_balancing,
                          test_size,
                          num_test_trials):
  """ Runs multiple version of random forest in parallel with selected features.

  The number of parallel processes are at most num_test_trials and never exceed
  number of available cores - 2.
  """
  finished_trials = 0
  metrics = list()
  while finished_trials < num_test_trials:
    num_processes = min(multiprocessing.cpu_count() - 2,
                        num_test_trials - finished_trials)
    print 'Running ' + str(num_processes) + ' parallel processes'
    # Replicate the data for processes
    replicated_inputs = ((features, labels, rf_num_trees, rf_criterion, rf_max_features,
                          rf_min_samples_split, rf_min_samples_leaf,
                          scikit_balancing, test_size),)*num_processes
    pool = multiprocessing.Pool(processes = num_processes)
    metrics.extend(pool.map(perform_single_random_forest, replicated_inputs))
    pool.close()
    finished_trials += num_processes

  # Take average of each metric
  mean_metrics = map(numpy.mean, zip(*metrics))
  print("Mean Test Accuracy: %0.2f%%\n" % mean_metrics[5])
  return mean_metrics


def perform_single_svm(input_data):
  """ Perform a single trial of svm with selected features
  """
  # extract inputs from input tuple
  features = input_data[0]
  labels = input_data[1]
  svm_kernel = input_data[2]
  svm_gamma = input_data[3]
  svm_cost = input_data[4]
  svm_degree = input_data[5]
  scikit_balancing = input_data[6]
  test_size = input_data[7]

  tolerance = 0.005
  cache_size = 6000

  # VERY IMPORTANT: Provide a random state, since it seems like multiple workers split the
  # data in the same way
  random.seed()
  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=test_size,
                                        random_state=random.randint(1,99999999)))

  train_features, train_labels, penalty_weights = utils.prepare_train_data(
      train_features, train_labels, scikit_balancing, -1)

  model = svm.SVC(tol=tolerance, cache_size=cache_size, class_weight=penalty_weights,
                  kernel = svm_kernel, gamma = svm_gamma,
                  C = svm_cost, degree = svm_degree)
  model = model.fit(train_features, train_labels)
  predicted_labels = model.predict(test_features)
  label_values = [0, 1]
  trial_metrics = compute_evaluation_metrics(predicted_labels, test_labels,
                                             label_values)
  return trial_metrics


def perform_svm(features,
                labels,
                svm_kernel,
                svm_gamma,
                svm_cost,
                svm_degree,
                scikit_balancing,
                test_size,
                num_test_trials):
  """ Runs multiple version of svm in parallel with selected features.

  The number of parallel processes are at most num_test_trials and never exceed
  number of available cores - 2.
  """
  finished_trials = 0
  metrics = list()
  while finished_trials < num_test_trials:
    num_processes = min(multiprocessing.cpu_count() - 2,
                        num_test_trials - finished_trials)
    print 'Running ' + str(num_processes) + ' parallel processes'
    # Replicate the data for processes
    replicated_inputs = ((features, labels, svm_kernel, svm_gamma, svm_cost, svm_degree,
                          scikit_balancing, test_size),)*num_processes
    pool = multiprocessing.Pool(processes = num_processes)
    metrics.extend(pool.map(perform_single_svm, replicated_inputs))
    pool.close()
    finished_trials += num_processes

  # Take average of each metric
  mean_metrics = map(numpy.mean, zip(*metrics))
  print("Mean Test Accuracy: %0.2f%%\n" % mean_metrics[5])
  return mean_metrics


def perform_single_logistic(input_data):
  """ Perform a single trial of logistic regression with selected features
  """
  # extract inputs from input tuple
  features = input_data[0]
  labels = input_data[1]
  logistic_penalty = input_data[2]
  logistic_cost = input_data[3]
  scikit_balancing = input_data[4]
  test_size = input_data[5]

  tolerance = 0.0005
  max_iterations = 10000

  # VERY IMPORTANT: Provide a random state, since it seems like multiple workers split the
  # data in the same way
  random.seed()
  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=test_size,
                                        random_state=random.randint(1,99999999)))

  train_features, train_labels, penalty_weights = utils.prepare_train_data(
      train_features, train_labels, scikit_balancing, -1)

  model = LogisticRegression(penalty=logistic_penalty, C=logistic_cost, tol=tolerance,
                             max_iter=max_iterations, class_weight=penalty_weights)
  model = model.fit(train_features, train_labels)
  predicted_labels = model.predict(test_features)
  label_values = [0, 1]
  trial_metrics = compute_evaluation_metrics(predicted_labels, test_labels,
                                             label_values)
  return trial_metrics


def perform_logistic(features,
                     labels,
                     logistic_penalty,
                     logistic_cost,
                     scikit_balancing,
                     test_size,
                     num_test_trials):
  """ Runs multiple version of logistic regression in parallel with selected features.

  The number of parallel processes are at most num_test_trials and never exceed
  number of available cores - 2.
  """
  finished_trials = 0
  metrics = list()
  while finished_trials < num_test_trials:
    num_processes = min(multiprocessing.cpu_count() - 2,
                        num_test_trials - finished_trials)
    print 'Running ' + str(num_processes) + ' parallel processes'
    # Replicate the data for processes
    replicated_inputs = ((features, labels, logistic_penalty, logistic_cost,
                          scikit_balancing, test_size),)*num_processes
    pool = multiprocessing.Pool(processes = num_processes)
    metrics.extend(pool.map(perform_single_logistic, replicated_inputs))
    pool.close()
    finished_trials += num_processes

  # Take average of each metric
  mean_metrics = map(numpy.mean, zip(*metrics))
  print("Mean Test Accuracy: %0.2f%%\n" % mean_metrics[5])
  return mean_metrics

    
def perform_single_knn(input_data):
  """ Perform a single trial of knn with selected features
  """
  # extract inputs from input tuple
  features = input_data[0]
  labels = input_data[1]
  knn_num_neighbors = input_data[2]
  knn_weights = input_data[3]
  knn_algorithm = input_data[4]
  knn_metric = input_data[5]
  knn_imbalanced_data = input_data[6]
  test_size = input_data[7]

  # VERY IMPORTANT: Provide a random state, since it seems like multiple workers split the
  # data in the same way
  random.seed()
  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=test_size,
                                        random_state=random.randint(1,99999999)))

  if not knn_imbalanced_data:
    # Manually balance data. Don't do this on the whole data set, instead do it only
    # on train set, so the it is balanced and precision per classes are equal.  Do not
    # balance test set, because test set should reflect the true distribution of new
    # points to predict.
    train_features, train_labels = utils.balance_data(train_features, train_labels)
  

  model = KNeighborsClassifier(n_neighbors=knn_num_neighbors, weights=knn_weights,
                               algorithm=knn_algorithm, metric=knn_metric)
  model = model.fit(train_features, train_labels)
  predicted_labels = model.predict(test_features)
  label_values = [0, 1]
  trial_metrics = compute_evaluation_metrics(predicted_labels, test_labels,
                                             label_values)
  return trial_metrics


def perform_knn(features,
                labels,
                knn_num_neighbors,
                knn_weights,
                knn_algorithm,
                knn_metric,
                knn_imbalanced_data,
                test_size,
                num_test_trials):
  """ Runs multiple version of knn in parallel with selected features.

  The number of parallel processes are at most num_test_trials and never exceed
  number of available cores - 2.
  """
  finished_trials = 0
  metrics = list()
  while finished_trials < num_test_trials:
    num_processes = min(multiprocessing.cpu_count() - 2,
                        num_test_trials - finished_trials)
    print 'Running ' + str(num_processes) + ' parallel processes'
    # Replicate the data for processes
    replicated_inputs = ((features, labels, knn_num_neighbors, knn_weights,
                          knn_algorithm, knn_metric, knn_imbalanced_data,
                          test_size),)*num_processes
    pool = multiprocessing.Pool(processes = num_processes)
    metrics.extend(pool.map(perform_single_knn, replicated_inputs))
    pool.close()
    finished_trials += num_processes

  # Take average of each metric
  mean_metrics = map(numpy.mean, zip(*metrics))
  print("Mean Test Accuracy: %0.2f%%\n" % mean_metrics[5])
  return mean_metrics


def main():
  df = pandas.read_csv(args.input_filename, index_col=False, header=0)
  data = df.values
  column_names = df.columns.values.tolist()
  
  # Impute the data and replace missing values
  imputer = preprocessing.Imputer(missing_values="NaN",
                                  strategy='mean',
                                  axis=0,
                                  copy=False)
  imputer.fit(data)
  data = imputer.transform(data)

  # Extract features/labels and their names from raw data
  features = data[:, 0:args.label_column]
  labels = data[:, args.label_column].astype(int)
  feature_names = column_names[0:args.label_column]
  label_name = column_names[args.label_column]

  # scale data no matter what, since the feature selector is L1-SVM
  (scaled_features, dummy) = utils.scale_data(features, None, 'minmax')

  # open output file and write header with max_num_features selected features
  output_file = open(args.output_filename, 'w')
  output_file_writer = csv.writer(output_file)
  header = ["num_features_selected",
            "test_size",
            "avg_true_positive", "avg_false_positive",
            "avg_true_negative", "avg_false_negative",
            "avg_accuracy",
            "avg_pos_f1" , "avg_neg_f1", "avg_average_f1",
            "avg_pos_precision", "avg_neg_precision", "avg_average_precision",
            "avg_pos_recall", "avg_neg_recall", "avg_average_recall"]

  for i in range(1, args.max_num_features+1):
    header.extend(["feature" + str(i), "feature" + str(i) + "_weight"])
  output_file_writer.writerow(header)

  feature_selector_obj = feature_selection.feature_selector(scaled_features,
                                                            labels,
                                                            args.num_samples,
                                                            args.scikit_balancing)

  for num_features in range(args.min_num_features, args.max_num_features+1):
    # Before anything, must set to feature selector object to num_feature
    feature_selector_obj.select_top_features(num_features)
    selected_features = feature_selector_obj.get_selected_features(feature_names)
    
    # Print selected and unselected features.
    print '\nSelected Feature,Weight'
    for feature, feature_coef in selected_features:
      print(feature + "," + str(feature_coef))

    # Now transform and restrict the features to those only selected by the L1-svm
    transformed_scaled_features = feature_selector_obj.transform(scaled_features)
    transformed_features = feature_selector_obj.transform(features)
    
    print('\n' + str(len(selected_features)) + ' out of ' + str(features.shape[1]) +
          ' features are selected.\n')
    
    # Now perform the learning task using the top features and report results. Make
    # sure to pass scaled features to svm
    num_test_trials = 10
    test_size = args.test_size if args.test_size <= 1.0 else int(args.test_size)
    if args.learning_algorithm == 'random-forest':
      rf_max_features = utils.extract_max_features(args.rf_max_features)
      metrics = perform_random_forest(transformed_features,
                                      labels,
                                      args.rf_num_trees,
                                      args.rf_criterion,
                                      rf_max_features,
                                      args.rf_min_samples_split,
                                      args.rf_min_samples_leaf,
                                      args.scikit_balancing,
                                      test_size,
                                      num_test_trials)

    elif args.learning_algorithm == 'svm':
      metrics = perform_svm(transformed_scaled_features,
                            labels,
                            args.svm_kernel,
                            args.svm_gamma,
                            args.svm_cost,
                            args.svm_degree,
                            args.scikit_balancing,
                            test_size,
                            num_test_trials)
    elif args.learning_algorithm == 'logistic':
      metrics = perform_logistic(transformed_features,
                                 labels,
                                 args.logistic_penalty,
                                 args.logistic_cost,
                                 args.scikit_balancing,
                                 test_size,
                                 num_test_trials)
    elif args.learning_algorithm == 'knn':
      metrics = perform_knn(transformed_scaled_features,
                            labels,
                            args.knn_num_neighbors,
                            args.knn_weights,
                            args.knn_algorithm,
                            args.knn_metric,
                            args.knn_imbalanced_data,
                            test_size,
                            num_test_trials)

    # write a row for num_features selected to output file
    output_row = [len(selected_features)]
    output_row.extend(metrics)
    for feature, feature_coef in selected_features:
      output_row.extend([feature, feature_coef])
    output_row.extend([''] * (len(header) - len(output_row)))
    output_file_writer.writerow(output_row)
    
    print '******************************\n'

  output_file.close()
  

if __name__ == "__main__":
  main()

