#!/usr/bin/python

import sys
import numpy
import argparse
import utils
import grid_search

from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(
    description='This script finds the best parameter set for various learning '
                'algorithms: svm, logistic regression, KNN and random forest.')
parser.add_argument('input_filename')
parser.add_argument('-a', '--algorithm', dest='algorithm',
                    default='kernel-svm',
                    choices=['kernel-svm', 'linear-svm', 'logistic', 'random-forest',
                             'knn'],
                    help = 'The choice of learning algorithm to perform grid search '
                    'for. The default value is svm. Other choices are knn, random-forest '
                    'and logistic regreesion.')
parser.add_argument('-s', '--scoring_functions', dest='scoring_functions',
                    default='accuracy',
                    help = 'The comma separated list of scoring functions used for cross '
                    'validation. Possible values are "accuracy", "weighted-precision", '
                    '"macro-precision", "weighted-recall", "macro-recall", '
                    '"weighted-f1" and "macro-f1". macro versions calculate the metric '
                    'for each label and report their unweighted mean. weighted version '
                    'reports the average, weighted by the number of true instances for '
                    'each label. Use macro if both labels should be treated the same '
                    'no matter the imbalance in the data set. Use weighted if you care '
                    'about overall precision, taking imbalance into account; for example '
                    'weighted would result in high reported precision if the majority '
                    'class has high precision while minority class has horrible '
                    'precision. For this reason, prefer macro, since imbalance in the '
                    'dataset does not really mean imbalance in reality. ')
parser.add_argument('-lc', '--label_column', dest='label_column',
                    type=int, default=94,
                    help='The column number of the label in the input csv. Defautl is '
                    '94, set it otherwise')
parser.add_argument('-nj', '--num_jobs', dest='num_jobs',
                    type=int, default=1,
                    help='Number of jobs for searching the grid parameter space in '
                    'parallel. Default is 1.')
parser.add_argument('-sb', '--scikit_balancing', dest='scikit_balancing',
                    default=False, action='store_true',
                    help='Whether to use scikit data balancing by changing penalties '
                    'in learning algorithm formulation or manually balance by '
                    'undersampling majority class and oversampling minority class')
args = parser.parse_args()

if __name__ == "__main__":
  # Let numpy know that NA corresponds to our missing value
  data=numpy.genfromtxt(args.input_filename, delimiter=",", skip_header=1,
                        missing_values="NA", filling_values = "NaN")
  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(data)
  data = imputer.transform(data)
  features = data[:, 0:args.label_column]
  labels = data[:, args.label_column].astype(int)

  train_features, test_features, train_labels, test_labels = (
      model_selection.train_test_split(features, labels, test_size=0.20))
  
  # scale data only if model is linear (svm, logisitic regression) or scales of features
  # are relevant (knn)
  if args.algorithm in ['linear-svm', 'kernel-svm', 'logistic', 'knn']:
    (train_features, test_features) = utils.scale_data(train_features,
                                                       test_features,
                                                       'minmax')

  # parse scoring methods
  scores = list()
  for scoring_function in args.scoring_functions.split(','):
    if not scoring_function in ["accuracy",
                                "weighted-precision",
                                "macro-precision",
                                "weighted-recall",
                                "macro-recall",
                                "weighted-f1",
                                "macro-f1"]:
      sys.exit('Invalid scoring function: ' + scoring_function + ' provided')
    scores.append(scoring_function)
 
  for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print("")
    
    clf = grid_search.grid_search(args.scoring_functions,
                                  train_features,
                                  train_labels,
                                  args.scikit_balancing,
                                  args.algorithm,
                                  args.num_jobs)

    y_true, y_pred = test_labels, clf.predict(test_features)
    labels = [0 , 1]
    target_names = ["female" , "male"]

    print("Best parameters set found on development set:")
    print("")
    print(clf.best_estimator_)
    print("")
    print("Grid scores on development set:")
    print("")
    for i in range(len(clf.cv_results_['params'])):
      print("{} (+/-{}) for{}".format(clf.cv_results_['mean_test_score'][i],
                                      clf.cv_results_['std_test_score'][i] / 2.0,
                                      clf.cv_results_['params'][i]))
      print("")

    print("Detailed classification report:")
    print("")
    print("The best model is trained on the balanced train data.")
    print("The scores below are computed for the best model on test set with original "
          "with its original distribution.")
    print("")
    print(classification_report(y_true, y_pred, labels, target_names))
