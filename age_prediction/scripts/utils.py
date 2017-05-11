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


def prepare_data(input_filename, label_column, train_size, test_size, add_log_vars):
  df = pd.read_csv(input_filename, delimiter=',', index_col=False, header=0) 
  data = df.values
  column_names = np.char.array(df.columns.values)
  print 'Number of columns in data {}'.format(len(column_names))

  # Extract features/labels and their names from raw data. Don't include the column next
  # to label, since it's gender
  features = data[:, 0:label_column-1]
  labels = data[:, label_column].astype(int)
  
  feature_names = column_names[0:label_column-1]
  label_name = column_names[label_column]

  class_values = list(set(labels))
  class_values.sort()

  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=test_size))
  
  # create requested train size.
  train_features, train_labels = undersample(train_features, train_labels, train_size)

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

  return (train_features, train_labels, test_features, test_labels,
          class_values, feature_names, label_name)


def write_data_predictions(output_filename, y_true, y_pred):
  data = np.column_stack((y_true, y_pred))

  # convert back to DF so that we can also write the column names
  df = pd.DataFrame(data, columns=['actual', 'predicted'])
  df.to_csv(output_filename, index=False)


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


def balance_data(features, labels, class_values, size=-1):
  """ transform the features and labels such that we have equal number of rows from each 
  class.
  
  size is the requested size per class
  """
  class_features = dict()
  class_labels = dict()
  class_sizes = dict()
  for val in class_values:
    class_indices = (labels[:] == val)
    class_features[val] = features[class_indices]
    class_labels[val] = labels[class_indices]
    class_sizes[val] = class_features[val].shape[0]
    
  
  if size < 0:
    size_per_class = np.min(class_sizes)
  else:
    size_per_class = size/len(class_values)
  
  if np.min(class_sizes.values()) < size_per_class:
    print('Not enough samples in a class {} vs required class size {}'.
          format(class_sizes, size_per_class))
    return (None, None)

  # now undersample each class 
  balanced_features = list()
  balanced_labels = list()
  for val in class_values:
    f, l = undersample(class_features[val], class_labels[val], size_per_class)
    balanced_features.append(f)
    balanced_labels.append(l)

  # concatenate both features and labels
  balanced_features = np.concatenate(balanced_features)
  balanced_labels = np.concatenate(balanced_labels)
  return (balanced_features, balanced_labels)
