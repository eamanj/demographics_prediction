#!/usr/bin/python

# This file include random forest utility functions to be used by other scripts.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler, Normalizer
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# function returns a custom callable scorer object to be used by grid search scoring. name
# identifies the type of the scorer to create. Possible names are accuracy,
# weighted-precision, macro-precision, weighted-recall, macro-recall, weighted-f1,
# macro-f1. Theses scorers are different from default version in that they are initialized
# with additional and non-default parameters.
def create_scorer(evaluation):
  if evaluation == "accuracy":
    return make_scorer(accuracy_score)
  elif evaluation == "weighted-precision":
    return make_scorer(precision_score, pos_label = None, average = 'weighted')
  elif evaluation == "macro-precision":
    return make_scorer(precision_score, pos_label = None, average = 'macro')
  elif evaluation == "weighted-recall":
    return make_scorer(recall_score, pos_label = None, average = 'weighted')
  elif evaluation == "macro-recall":
    return make_scorer(recall_score, pos_label = None, average = 'macro')
  elif evaluation == "weighted-f1":
    return make_scorer(f1_score, pos_label = None, average = 'weighted')
  elif evaluation == "macro-f1":
    return make_scorer(f1_score, pos_label = None, average = 'macro')
  elif evaluation == "mae":
    return make_scorer(mean_absolute_error, multioutput='uniform_average')
  elif evaluation == "mse":
    return make_scorer(mean_squared_error, multioutput='uniform_average')
  else:
    sys.exit('Invalid scoring function: ' + scoring_function + ' provided')


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
  
  indices = np.random.choice(features.shape[0], size, replace=False)
  undersampled_features = features[indices, :]
  undersampled_labels = labels[indices] 
  return (undersampled_features, undersampled_labels)


def oversample(features, labels, size):
  num_extra_samples = size - features.shape[0]
  # make sure we use all existing rows at least once
  indices = np.array(range(0, features.shape[0]))
  # now append num_extra_samples random indices to the list
  extra_indices = np.random.randint(0, features.shape[0], num_extra_samples)
  indices = np.append(indices, extra_indices)
  oversampled_features = features[indices, :]
  oversample_labels = labels[indices]
  return (oversampled_features, oversample_labels)

def balance_data(features, labels, class_values, size=-1):
  """ transform the features and labels such that we have equal number of rows from each 
  class.
  
  size is the requested size per class
  The large classes are undersampled until its size matches the minority class.
  if size is negative, all of minority class will be included and majority class will be
  undersampled to match it. In other words, the size of output becomes num classes times
  the size of minority class.
  """
  # a dict from label value to its size in the data set
  num_labels = dict()
  for class_value in class_values:
    class_indices = (labels[:] == class_value)
    class_features = features[class_indices]
    num_labels[class_value] = class_features.shape[0]

  if size < 0:
    size_per_class = min(num_labels.values())
  else:
    size_per_class = size/len(class_values)

  balanced_features_list = list()
  balanced_classes_list = list()
  for class_value in class_values:
    class_indices = (labels[:] == class_value)
    class_features = features[class_indices]
    class_labels = labels[class_indices]
    if num_labels[class_value] < size_per_class:
      print('WARNING: NOT enough ' + str(class_value) + ' samples: ' +
            str(num_labels[class_value]) + '. Will oversample.')
      balanced_class_features, balanced_class_labels = oversample(class_features,
                                                                  class_labels,
                                                                  size_per_class)
    else:
      balanced_class_features, balanced_class_labels = undersample(class_features,
                                                                   class_labels,
                                                                   size_per_class)
    balanced_features_list.append(balanced_class_features)
    balanced_classes_list.append(balanced_class_labels)

  # concatenate all features and labels
  balanced_features = np.concatenate(balanced_features_list)
  balanced_labels = np.concatenate(balanced_classes_list)
  return (balanced_features, balanced_labels)


def prepare_train_data(train_features, train_labels, imbalanced_data, class_values,
                       train_size):
  """ Balances the data set in two possible ways. Scales it to requested size
  """
 
  if imbalanced_data:
    # Automatically adjust parameter C of each class inversely proportional to class
    # frequencies
    train_features, train_labels = undersample(train_features, train_labels, train_size)
  else:
    # Manually balance data. Don't do this on the whole data set, instead do it
    # only on train set, so the it is balanced and precision per classes are equal.
    # Do not balance test set, because test set should reflect the true distribution of
    # new points to predict.
    train_features, train_labels = balance_data(train_features, train_labels,
                                                class_values, train_size)

  return (train_features, train_labels)


def prepare_data(input_filename, label_column, train_size, test_size, imbalanced_data,
                 add_log_vars):
  """ Reads the data from input file, separates features from label and generates the
  train and test set matching their requested size. It also balances the train set if
  requested (imbalanced_data is false). Finally it imputes both train and test features.

  return imputed train_features, train_labels, imputed test features, test labels,
  class_values: set of unique class values
  class_names: set of unique class/target names (string)
  """
  df = pd.read_csv(input_filename, delimiter=',', index_col=False, header=0) 
  data = df.values
  column_names = np.char.array(df.columns.values)
  print 'Number of columns in data {}'.format(len(column_names))

  # Extract features/labels and their names from raw data
  features = data[:, 0:label_column]
  labels = data[:, label_column].astype(int)
  
  feature_names = column_names[0:label_column]
  label_name = column_names[label_column]

  # balance it
  if True:
    np.place(labels, np.logical_or(labels == 1, labels == 2), 2)
    np.place(labels, np.logical_or(labels == 5, labels == 6), 5)
    np.place(labels, labels > 6, 6)

  class_values = list(set(labels))
  class_values.sort()
  class_names = map(str, class_values)
  print 'Counts per classes are {}'.format(np.unique(labels, return_counts=True)[1])

  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=test_size))
  
  # balance the train data set and create requested train size.
  train_features, train_labels = prepare_train_data(train_features,
                                                    train_labels,
                                                    imbalanced_data,
                                                    class_values,
                                                    train_size)
  if not np.all(np.unique(train_labels) == np.unique(labels)):
    print 'unique labels in train ' + str(np.all(np.unique(train_labels)))
    print 'unique labels ' + str(np.unique(labels))
    sys.exit('we need the train set to have all different classes')
 
  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(train_features)
  train_features = imputer.transform(train_features)
  test_features = imputer.transform(test_features)
 
  # Only after imputing nans, get list of columns with negative values, so we won't apply
  # log-transformation on them
  if add_log_vars:
    column_mins = np.amin(np.concatenate((train_features, test_features), axis=0), axis=0)
    pos_feature_names = feature_names[column_mins>=0]
    neg_feature_names = feature_names[column_mins<0]
    pos_train_features = train_features[:,column_mins>=0]
    pos_test_features = test_features[:,column_mins>=0]
    # make sure negative features are only skewness related
    assert all(['skewness' in feature for feature in neg_feature_names])
    
    # add logof(plus-one) version to features
    transformer = FunctionTransformer(np.log1p)
    log_pos_train_features = transformer.transform(pos_train_features)
    log_pos_test_features = transformer.transform(pos_test_features)
    log_pos_feature_names = pos_feature_names + "_log"
    train_features = np.concatenate((train_features, log_pos_train_features), axis=1)
    test_features = np.concatenate((test_features, log_pos_test_features), axis=1)
    feature_names = np.concatenate((feature_names, log_pos_feature_names))
    print 'Number of columns in data after adding log vars {}'.format(len(feature_names))

  feature_label_names = np.append(feature_names, label_name)
  return (train_features, train_labels, test_features, test_labels,
          class_values, class_names, feature_label_names)


def scale_data(train_features, test_features, scaling_method):
  if scaling_method:
    if scaling_method == 'minmax':
      scaler = MinMaxScaler(feature_range = (0, 1), copy = False)
    elif scaling_method == 'normal':
      scaler = Normalizer(norm = 'l2', copy = False)
    elif scaling_method == 'standard':
      scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    elif scaling_method is not None:
      sys.exit('Invalid scaler...')
    train_features = scaler.fit_transform(train_features)
    if test_features is not None:
      test_features = scaler.transform(test_features)
  return (train_features, test_features)


def plot_confusion_matrix(cm, class_names, title='Confusion matrix', cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
