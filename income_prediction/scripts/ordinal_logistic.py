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
  description='Ordinal Logistic regression script.')
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
parser.add_argument('-s', '--scaler', dest='scaling_method',
                    choices=['normal', 'standard', 'minmax'], default=None,
                    help='The type of scaling method to use. Possible options are: '
                    'normal, standard, minmax. Default option does not scale. '
                    'normal: normalizes the features based on L2 norm. '
                    'standard: standardizes features so that they are zero centered and '
                    'have unit variance. '
                    'minmax: scales the features based on minimum and maximum so that '
                    'they are between 0 and 1.')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=1257,
                    help='The column number of the label in the input csv. Default is '
                    '1257 starting from zero, set it otherwise')
parser.add_argument('-oa', '--oridinal_algorithm', dest='ordinal_algorithm',
                    choices = ['logisticse' , 'logisticit', 'logisticat',
                               'ordinalridge', 'lad'], default='logisticat',
                    help='The type of multiclass algorithm to use. The choices are '
                    'Squared Error variation (se), Immediate Threshold variant (IT), '
                    'All Threshold variant (at), Ordinal Rridge (ridge) and Least '
                    'Absolute Deviation (lad). Default is multi.')
parser.add_argument('-fs', '--feature_selection_algo', dest='feature_selection_algo',
                    choices = ['anova', 'best'], default=None,
                    help = 'Determines how to perform feature selection. Choices are '
                    'f-classif (anova), and best which selects '
                    'either that achieves higheset performance in terms of evaluation. '
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
parser.add_argument('-a', '--alpha', dest='alpha', default=1, type=float,
                    help='The alpha parameter to use only if grid search is skipped. '
                    'Only applicable if using LogisticIT, LogisticAT, LogisticSE and '
                    'OrdinalRdige.')
parser.add_argument('-c', '--cost', dest='cost', default=1.0, type=float,
                    help='The cost regualarization parameter to use only if grid search '
                    'is skipped. Only applicable if using LAD.')
parser.add_argument('-l', '--loss', dest='loss',
                    default='l1', choices=['l1', 'l2'],
                    help='The loss function to only if grid search is skipped. Only '
                    'applicable if using LAD. choices are l1 and l2. Default is l1.')
args = parser.parse_args()

def main():
  (train_features, train_labels, test_features, test_labels, class_values, class_names,
   feature_label_names) = utils.prepare_data(args.input_filename,
                                             args.label_column,
                                             args.train_size,
                                             args.test_size,
                                             args.imbalanced_data)
  # now that we have limited the data to requested train size, scale data since svm needs
  # to be scaled
  (train_feautres, test_features) = utils.scale_data(train_features,
                                                     test_features,
                                                     args.scaling_method)
  
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
 
  # unfortunately, the current implementation does not handle weights by class frequency.
  # so we don't pass penalty weights
  model = models.train_ordinal_logistic(train_features,
                                        train_labels,
                                        args.skip_grid_search,
                                        args.evaluation,
                                        args.num_jobs,
                                        args.loss,
                                        args.alpha,
                                        args.cost,
                                        args.ordinal_algorithm)

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
