#!/usr/bin/python

import argparse
import numpy as np
import pandas as pd
import scipy.stats.mstats
import utils
import models
import feature_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


parser = argparse.ArgumentParser(
  description='Multinomial/OVR classification using logistic regression script.')
parser.add_argument('input_filename')
parser.add_argument('output_filename')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=1384,
                    help='The column number of the label in the input csv. Default is '
                    '1384 starting from zero, set it otherwise')
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
parser.add_argument('-nc', '--num_classes', dest='num_classes',
                    type=int, default=4,
                    help='The number of equal sized age classes to create from the '
                    'nominal age values.')
parser.add_argument('-p', '--penalty', dest='penalty',
                    choices= ['l1', 'l2'], default='l2',
                    help='The norm to use for the penalty. Must be l1 or l2. Note that '
                    'Cross validation wont optimize this; so you need to test both '
                    'l1 and l2 if you want the optimal results. Has to be l2 if in '
                    'multinomial mode')
parser.add_argument('-m', '--multi_class', dest='multi_class',
                    choices = ['ovr', 'multinomial'], default='ovr',
                    help='Determines how to solve multi classification. Has to be '
                    'eihter one-vs-rest or multinomial. ovr is default.')
parser.add_argument('-nj', '--num_jobs', dest='num_jobs', type=int, default=-1,
                    help='Number of jobs to instantiate for cross validation. Default is '
                    '-1, which corresponds to the number of cores in the machine')

# feature selection flags
parser.add_argument('-sf', '--skip_feature_selection', dest='skip_feature_selection',
                    default=False, action='store_true',
                    help='If specified, skips feature selection. Default is to run '
                    'cross validated feature selection on original features and '
                    'transform data into a smaller dimension.')
parser.add_argument('-fc', '--feature_selection_cost', dest='feature_selection_cost',
                    default=None, type=float,
                    help='The cost of L1-svm feature selection. If not provided, the '
                    'best cost of the l1-svm will be determined through grid search. '
                    'Default is None, which leads to grid search of best cost. Relevent '
                    'only if feature selection is not skipped.')
parser.add_argument('-t', '--threshold', dest='threshold', default=0, type=float,
                    help='Feature selection threshold to elastic net. If not specified '
                    'all features with non-zero importance by elastic net will be used.')

# cross validation flags: they will be ignored if cross validation is not skipped.
parser.add_argument('-sc', '--skip_cross_validation', dest='skip_cross_validation',
                    default=False, action='store_true',
                    help='If specified, skips cross validation. Default is to run cross '
                    'validation with cross validation to determine the best parameters. '
                    'Then run the training once more with best params. If specified, '
                    'this step is skipped, and the provided parameters are used to train '
                    'the model. Default is False which performs the cross validation.')
parser.add_argument('-c', '--cost', dest='cost', default=1.0, type=float,
                    help='Inverse of the regularization strength; must be a positive '
                    'float. Like in support vector machines, smaller values mean '
                    'stronger regularization. Relevent only if cross validation is '
                    'skipped.')
parser.add_argument('-e', '--evaluation', dest='evaluation',
                    choices = ["accuracy", "weighted-precision", "macro-precision",
                               "weighted-recall", "macro-recall",
                               "weighted-f1", "macro-f1",
                               "mae", "mse"],
                    default = "macro-f1",
                    help = 'The evaluation metric to use in cross validation. Note that '
                    'mae and mse are regression metrics that treat the outcome variable '
                    'as an ordinal variable rather than a categorical variable.')
args = parser.parse_args()


def main():
  # repeat and get more samples than needed until we can create fully balanced test and
  # train. We might not be since labels falling on the quantiles get assigned to the left
  # class, thus making the sample unbalanced
  oversample_rate = 1.1
  while True:
    add_log_vars = True
    (train_features, train_labels, test_features, test_labels, class_values,
     feature_names, label_name) = utils.prepare_data(args.input_filename,
                                                     args.label_column,
                                                     int(args.train_size*oversample_rate),
                                                     int(args.test_size*oversample_rate),
                                                     add_log_vars)
    all_labels = np.concatenate([train_labels, test_labels])
    
    # change labels to equally sized classes, using all labels in case minimum falls in
    # test.
    quantiles = scipy.stats.mstats.mquantiles(all_labels,
                                              np.arange(0, 1, 1.0/args.num_classes))
    train_labels = np.digitize(train_labels, quantiles)
    test_labels = np.digitize(test_labels, quantiles)
    class_values = range(1, args.num_classes + 1)
    train_features, train_labels = utils.balance_data(train_features, train_labels,
                                                      class_values, args.train_size)
    test_features, test_labels = utils.balance_data(test_features, test_labels,
                                                    class_values, args.test_size)
    if train_features is None or test_features is None:
      oversample_rate += 0.1
      continue
    break
  
  all_labels = np.concatenate([train_labels, test_labels])
  print("Label is {}".format(label_name))
  print("Max of all labels: %d" % np.max(all_labels))
  print("Min of all labels: %d" % np.min(all_labels))

  if not args.skip_feature_selection:
    (train_features, test_features, feature_names) = feature_selection.L1SVMFeatureSelection(
        train_features, train_labels, test_features, feature_names,
        args.feature_selection_cost, args.threshold, args.num_jobs)

  model = models.train_logistic(train_features,
                                train_labels,
                                args.skip_cross_validation,
                                args.multi_class,
                                args.penalty,
                                args.evaluation,
                                args.num_jobs,
                                args.cost)

  # Predict test and report full stats
  y_true = test_labels
  y_pred_prob = model.predict_proba(test_features)
  df = pd.DataFrame(data=y_pred_prob, columns=model.classes_)
  df['max_prob'] = df.max(axis=1)
  df['max_prob_class'] = df.idxmax(axis=1)
  df['true'] = y_true
  y_pred = df['max_prob_class']
  # TODO: if requested, choose the predicted values such that the class frquencies match
  # the expected class frequencies

  print("\n*****************************\n")
  print('MAE on test: {}'.format(
      mean_absolute_error(y_true, y_pred, multioutput='uniform_average')))
  
  print('Test Accuracy: {}'.format(accuracy_score(y_true, y_pred)*100.))
  print('Classification report:')
  print(classification_report(y_true, y_pred, class_values))
  print('Weighted Precision Recall:')
  print(precision_recall_fscore_support(y_true, y_pred, labels=class_values,
                                        pos_label=None,
                                        average='weighted'))
  print('Unweighted Precision Recall:')
  print(precision_recall_fscore_support(y_true, y_pred, labels=class_values,
                                        pos_label=None,
                                        average='macro'))
  
  # print and plot confusion matrix
  print('Confusion Matrix Without Normalization')
  np.set_printoptions(precision=2)
  cm = confusion_matrix(y_true, y_pred, class_values)
  print(cm)
  print('Confusion Matrix With Normalization')
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print(cm_normalized)
  
  plt.figure()
  plt.subplot(2, 1, 1)
  utils.plot_confusion_matrix(cm, class_values, 'Unnormalized confusion matrix')

  # Normalize the confusion matrix by row (i.e by the number of samples
  # in each class)
  plt.subplot(2, 1, 2)
  utils.plot_confusion_matrix(cm_normalized, class_values, 'Normalized confusion matrix')

  pdf = PdfPages(args.output_filename + '.pdf')
  plt.savefig(pdf, format='pdf')
  pdf.close()
  
  # Now print stats on subsets based on confidence of max_prob_class. Sort predictions
  # by confidence in descending order and take subsets from the top of the sorted df
  df = df.sort_values(by='max_prob', ascending=False)
  print(','.join(['Probability Threshold', 'Percentage Predicted', 'Accuracy',
                  'AverageRecall', 'AveragePrecision', 'AverageFscore']))
  for percent_to_predict in range(1,100):
    lowest_idx = int(percent_to_predict * len(df.index) / 100.0)
    df_subset = df.iloc[0:lowest_idx]
    prob_threshold = df_subset['max_prob'].min()
    accuracy = accuracy_score(df_subset['true'], df_subset['max_prob_class'])
   
    (precision, recall, fscore, support) = precision_recall_fscore_support(
        y_true, y_pred, labels=class_values, pos_label=None, average='macro')
    print(','.join(map(str, [prob_threshold, percent_to_predict, accuracy,
                             recall, precision, fscore])))


if __name__ == '__main__':
  main()
