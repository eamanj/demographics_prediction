#!/usr/bin/python

# This file include random forest utility functions to be used by other scripts.

import scipy.stats
import sys
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# The representation of male and female in cleaned input file
FEMALE = 0
MALE = 1

# Extracts the appropriate value of max_features to be passed to RF instance.
# It can handle strings, floats and integers and will interpret each
# differently.
def extract_max_features(max_features):
  if max_features == "all":
    return None
 
  result = max_features
  # is it a float?
  try:
    result = float(max_features)
    # yes it is. must be less than 1
    if result > 1.0 or result < 0:
      sys.exit("Bad max_features. Must be between 0 and 1.")
  except ValueError:
    pass

  # is it an integer
  try:
    result = int(max_features)
    # TODO: check if result is less than number of features
  except ValueError:
    pass
  return result

# Compute sample weights such that the class distribution of y becomes balanced. But it
# seems like random forest or at least scikit implementation of random forest does not
# balance the weights, because the results of training in a balance data set is very
# different from training on an imbalanced data set (i.e. the accuracy stays the same, but
# the minority class has a very low recall in the presence of data imbalance, even though
# we explicitly asked scikit to adjust class weights to account for imbalance)
def balance_weights(y):
  y = numpy.asarray(y)
  y = numpy.searchsorted(numpy.unique(y), y)
  bins = numpy.bincount(y)

  weights = 1. / bins.take(y)
  weights *= bins.min()
  return weights


def undersample(features, labels, size):
  """ Returns an undersampled version of features.
  
  The returned 2d array contains size random rows from the original features
  without replacement. The idea is we want to keep the undersampled version as diverse
  as possible.
  if size is negative, no undersampling takes place
  """
  if size < 0:
    return (features, labels)

  num_samples = features.shape[0]
  if num_samples < size:
    sys.exit('Not enough samples: ' + str(num_samples))
  
  indices = numpy.random.choice(features.shape[0], size, replace=False)
  undersampled_features = features[indices, :]
  undersampled_labels = labels[indices] 
  return (undersampled_features, undersampled_labels)


def balance_data(features, labels, size=-1):
  """ transform the features and labels such that we have equal number of rows from each 
  class.
  
  size is the requested size per class
  First the minority class is over sampled to match the requested size.
  If there is still imbalance, the majority class is undersampled until its size matches
  the minority class.
  if size is negative, all of minority class will be included and majority class will be
  undersampled to match it. In other words, the size of output becomes twice the size of
  minority class.
  """
  male = 1
  female = 0
  male_indices = (labels[:] == male)
  female_indices = (labels[:] == female)
 
  male_features = features[male_indices]
  female_features = features[female_indices]
  male_labels = labels[male_indices]
  female_labels = labels[female_indices]

  num_males = male_features.shape[0]
  num_females = female_features.shape[0]
  if size < 0:
    size_per_class = min(num_males, num_females)
  else:
    size_per_class = size/2

  if num_males < size_per_class:
    sys.exit('Not enough male samples: ' + str(num_males))
  if num_females < size_per_class:
    sys.exit('Not enough female samples: ' + str(num_females))

  male_features, male_labels = undersample(male_features, male_labels,
                                           size_per_class)
  female_features, female_labels = undersample(female_features, female_labels,
                                               size_per_class)
  # concatenate both features and labels
  balanced_features = numpy.concatenate((female_features, male_features))
  balanced_labels = numpy.concatenate((female_labels, male_labels))
  return (balanced_features, balanced_labels)


def prepare_train_data(train_features, train_labels, scikit_balancing, train_size):
  """ Balances the data set in two possible ways. Scales it to requested size
  """
 
  if scikit_balancing:
    # Automatically adjust parameter C of each class inversely proportional to class
    # frequencies
    penalty_weights = 'auto'
    train_features, train_labels = undersample(train_features, train_labels, train_size)
  else:
    # Manually balance data. Don't do this on the whole data set, instead do it
    # only on train set, so the it is balanced and precision per classes are equal.
    # Do not balance test set, because test set should reflect the true distribution of
    # new points to predict.
    train_features, train_labels = balance_data(train_features, train_labels, train_size)
    penalty_weights = None

  return (train_features, train_labels, penalty_weights)


# Given the predicted probabilities from RF trees or SVM margin, computes the probability
# threshold that would produce the desired percentage of *best* data to predict. Best here
# means "most predictable" or those points whose probabiity is above the threshold.
def find_matching_probability_threshold(predicted_probabilities, percentage_to_predict):
  probability_levels = 100000
  probability_level_count = [0] * (probability_levels+1)
  num_all_points = 0
  for probabilities in predicted_probabilities:
    positive_probability = int(probabilities[0] * probability_levels)
    negative_probability = int(probabilities[1] * probability_levels)

    probability_level_count[max(positive_probability, negative_probability)] += 1
    num_all_points += 1

  # find the level whose cumulative sum matches the desired percentage
  num_points_so_far = 0
  for threshold in range(probability_levels, 0, -1):
    num_points_so_far += probability_level_count[threshold-1]
    if num_points_so_far * 100.0 / num_all_points > percentage_to_predict:
      return threshold * 100.0 / probability_levels

  # if you are here, you never matched the desired percentage, return minimum possible
  # threshold
  return 50.0

def compute_threshold_metrics(threshold, predicted_probabilities,
                              test_labels, label_values):
  """
  Computes various evaluation metrics at a fixed score threshold, when predictions are
  probabilities.

  Returns a tuple containing:
  (accuracy, percentage predicted, true_positive, false_negative, true_positive_rate,
  false_negative_rate, true_negative, false_positive, true_negative_rate,
  false_positive_rate)
  at the probability threshold. The idea is we only make a prediction if the predicted
  probability exceeds the threshold.

  threshold: only scores above this threshold in [0,100] are included in evaluation
  predicted_probabilities: the score probability for each test input. a score between 0
  and 1.
  test_labels: contains the actual labels per each data point. The order of labels in this
  list must correspond to the order of probabilities in predicted_probabilities.
  label_values: contains the value of each label;
  """
  assert numpy.shape(predicted_probabilities)[0] == len(test_labels)
  assert numpy.shape(predicted_probabilities)[1] == 2

  true_positive = 0
  false_positive = 0
  true_negative = 0
  false_negative = 0
  missed_positive = 0
  missed_negative = 0

  for i in range(0, len(test_labels), 1):
    positive = label_values[0]
    positive_probability = predicted_probabilities[i][0] * 100
    negative = label_values[1]
    negative_probability = predicted_probabilities[i][1] * 100
    # Should we predict label 1?
    if positive_probability > threshold:
      # is our prediction correct?
      if test_labels[i] == positive:
        true_positive += 1
      else:
        false_positive += 1
    # Should we predict label 2?
    elif negative_probability >= threshold:
      # is our prediction correct?
      if test_labels[i] == negative:
        true_negative += 1
      else:
        false_negative += 1
    # We refuse to predict this point.
    else:
      if test_labels[i] == positive:
        missed_positive += 1
      else:
        missed_negative += 1

  total_predicted = true_positive + true_negative + false_positive + false_negative
  accuracy = (true_positive + true_negative) * 100.0 / total_predicted
  percentage_predicted = (
      total_predicted * 100.0 / (total_predicted + missed_positive + missed_negative))

  true_positive_rate = true_positive * 100.0 / (true_positive + false_negative)
  true_negative_rate = true_negative * 100.0 / (true_negative + false_positive)
  false_positive_rate = 100 - true_negative_rate
  false_negative_rate = 100 - true_positive_rate

  return (accuracy, percentage_predicted,
          true_positive, false_negative, true_positive_rate, false_negative_rate,
          true_negative, false_positive, true_negative_rate, false_positive_rate)

def print_threshold_metrics(predicted_probabilities, test_labels, label_values,
                            low=4, high=100):
  """
  prints evaluation of test data at different probability thresholds
  i.e. reports accuracy only for points whose predicted probability
  for either label is above the specified threshold
  """

  # header for evaluation of metrics per threshold
  print("\n*****************************\n")

  # TODO isn't this the opposite?
  print 'Positive is female (0).'
  print 'Negative is male (1).'

  print('Probability Threshold,Accuracy,Percentage Predicted,'
        'True Positive,False Negative,True Positive Rate,'
        'False Negative Rate,True Negative,False Positive,'
        'True Negative Rate,False Positive Rate')

  for percentage_to_predict in range(high, low, -1):
    # find probability threshold that matches our desired
    # percentage_to_predict
    threshold = find_matching_probability_threshold(
        predicted_probabilities, percentage_to_predict)
    evaluation = compute_threshold_metrics(threshold,
        predicted_probabilities, test_labels, label_values)
    evaluation = (threshold,) + evaluation

    print('%0.2f,%0.2f,%0.2f,%d,%d,%0.2f,%0.2f,%d,%d,%0.2f,%0.2f' %
          evaluation)


def scale_data(train_features, test_features, scaling_method,
               minmax_min = 0, minmax_max = 1):
  if scaling_method:
    if scaling_method == 'minmax':
      scaler = MinMaxScaler(feature_range = (minmax_min, minmax_max), copy = True)
    elif scaling_method == 'normal':
      scaler = Normalizer(norm = 'l2', copy = True)
    elif scaling_method == 'standard':
      scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    elif scaling_method is not None:
      sys.exit('Invalid scaler...')
    scaled_train_features = scaler.fit_transform(train_features)

    # have to scale test using the learned scaler only on the availabe train data size
    scaled_test_features = None
    if test_features is not None:
      scaled_test_features = scaler.transform(test_features)
  return (scaled_train_features, scaled_test_features)


def write_data_predictions(output_filename_prefix, features,
                           feature_names, y_true, y_pred):
  male_output_filename = output_filename_prefix + '.males'
  female_output_filename = output_filename_prefix + '.females'
  male_indices = (y_true[:] == MALE)
  female_indices = (y_true[:] == FEMALE)
  male_features = features[male_indices]
  female_features = features[female_indices]
  male_correct_pred = (y_pred[male_indices] == MALE).astype(int)
  female_correct_pred = (y_pred[female_indices] == FEMALE).astype(int)
  male_data = numpy.column_stack((male_features, male_correct_pred))
  female_data = numpy.column_stack((female_features, female_correct_pred))

  # convert back to DF so that we can also write the column names
  male_df = pandas.DataFrame(male_data, columns=(feature_names + ['correct_prediction']))
  male_df.to_csv(male_output_filename, index=False)
  female_df = pandas.DataFrame(female_data, columns=(feature_names + ['correct_prediction']))
  female_df.to_csv(female_output_filename, index=False)
