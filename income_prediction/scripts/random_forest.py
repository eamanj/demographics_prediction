#!/usr/bin/python

import argparse
import numpy
import utils
import feature_selection
import models
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(
  description='Random Forest script.')
parser.add_argument('input_filename')
parser.add_argument('output_figure')
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
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=1257,
                    help='The column number of the label in the input csv. Default is '
                    '1257 starting from zero, set it otherwise')
parser.add_argument('-fs', '--feature_selection_algo', dest='feature_selection_algo',
                    choices = ['f-classif', 'l1-svm'], default=None,
                    help = 'Determines how to perform feature selection. Choices are '
                    'f-classif (anova), and l1-svm (lasso svm). '
                    'The default is None, which performs no feature selection. So '
                    'if you dont want any feature selection, dont set this flag.')
parser.add_argument('-sg', '--skip_grid_search', dest='skip_grid_search',
                    default=False, action='store_true',
                    help='If specified, skips grid search. Default is to run cross '
                    'validation with grid search to determine the best parameters. Then '
                    'run the training once more with best params. If specified, this '
                    'step is skipped, and the provided parameters are used to train the '
                    'model. Default is False which performs the grid search.')
parser.add_argument('-i', '--imbalanced_data', dest='imbalanced_data',
                    default=False, action='store_true',
                    help='Whether to use balance the data in terms of their labels '
                    'training or use the imbalance data. default is False which balances '
                    'the data.')
parser.add_argument('-e', '--evaluation', dest='evaluation',
                    choices = ["accuracy", "weighted-precision", "macro-precision",
                               "weighted-recall", "macro-recall",
                               "weighted-f1", "macro-f1",
                               "mae", "mse"],
                    default = "macro-f1",
                    help = 'The evaluation metric to use in grid search. Note that '
                    'mae and mse are regression metrics that treat the outcome variable '
                    'as an ordinal variable rather than a categorical variable. So '
                    'scikit balancing is irrelevant when evaluation is mae or mse.')
parser.add_argument('-nj', '--num_jobs', dest='num_jobs', type=int, default=-1,
                    help='Number of jobs to instantiate for grid search. Default is '
                    '-1, which corresponds to the number of cores in the machine')

# The following will be ignored if grid search is not skipped.
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


def print_feature_importances(model, feature_label_names, num_top_features):
  importances = model.feature_importances_
  indices = numpy.argsort(importances)[::-1]
  print("Feature ranking:")
  for f in range(num_top_features):
    print("%d. %s (%f)" % (f + 1, feature_label_names[indices[f]],
                           importances[indices[f]]))


def main():
  (train_features, train_labels, test_features, test_labels, class_values, class_names,
   feature_label_names) = utils.prepare_data(args.input_filename,
                                             args.label_column,
                                             args.train_size,
                                             args.test_size,
                                             args.imbalanced_data)

  # We let scikit use its balancing scheme if it is explicitly requested
  penalty_weights = 'balanced' if args.imbalanced_data else None
 
  # feature selection if requested
  if args.feature_selection_algo:
    feature_selector_obj =  feature_selection.feature_selector(args.evaluation,
                                                               train_features,
                                                               train_labels,
                                                               feature_label_names,
                                                               -1,
                                                               penalty_weights,
                                                               args.feature_selection_algo,
                                                               args.num_jobs)
    train_features = feature_selector_obj.transform(train_features)
    test_features = feature_selector_obj.transform(test_features)
    print "Selected " + str(len(feature_selector_obj.get_selected_features())) + " features"
    print "Top 10 features: " + str(feature_selector_obj.get_top_features(10))
 

  max_features = extract_max_features(args.max_features)
  model = models.train_random_forest(train_features,
                                     train_labels,
                                     penalty_weights,
                                     args.skip_grid_search,
                                     args.evaluation,
                                     args.num_jobs,
                                     args.num_trees,
                                     args.criterion,
                                     max_features,
                                     args.min_samples_split,
                                     args.min_samples_leaf)

  # Predict test and report full stats
  y_true, y_pred = test_labels, model.predict(test_features)
  print("\n*****************************\n")
  print('MAE: ' +
        str(metrics.mean_absolute_error(y_true, y_pred, multioutput='uniform_average')))
  print('MSE: ' +
        str(metrics.mean_squared_error(y_true, y_pred, multioutput='uniform_average')))
  
  print('Classification report:')
  print(metrics.classification_report(y_true, y_pred, class_values, class_names))
  print('Precision Recall')
  print(metrics.precision_recall_fscore_support(y_true, y_pred, labels=class_values,
                                                pos_label=None,
                                                average='weighted'))

  # print and plot confusion matrix
  print('Confusion Matrix Without Normalization')
  numpy.set_printoptions(precision=2)
  cm = metrics.confusion_matrix(y_true, y_pred, class_values)
  print(cm)
  print('Confusion Matrix With Normalization')
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
  print(cm_normalized)
 
  # print top 40 features
  print("\n*****************************\n")
  print("Features importances: ")
  print_feature_importances(model, feature_label_names, 40)
  
  plt.figure()
  plt.subplot(2, 1, 1)
  utils.plot_confusion_matrix(cm, class_names, 'Unnormalized confusion matrix')

  # Normalize the confusion matrix by row (i.e by the number of samples
  # in each class)
  plt.subplot(2, 1, 2)
  utils.plot_confusion_matrix(cm_normalized, class_names, 'Normalized confusion matrix')

  #plt.savefig(args.output_figure + '.pdf')
  pdf = PdfPages(args.output_figure + '.pdf')
  plt.savefig(pdf, format='pdf')
  pdf.close()
  
if __name__ == '__main__':
  main()

