#!/usr/bin/python

import argparse
import numpy
import pandas
import utils
from sklearn.preprocessing import Imputer

parser = argparse.ArgumentParser(
  description='Simple script for imputing missing column data, replacing them with '
              'column averages. You can also scale the features using different scalers.')
parser.add_argument('input_filename')
parser.add_argument('output_filename')
parser.add_argument('-s', '--scaler', dest='scaling_method',
                    choices=['normal', 'standard', 'minmax', None], default=None,
                    help = 'The type of scaling method to use. Possible options are: '
                    'normal, standard, minmax or unspecified'
                    'normal: normalizes the features based on L2 norm. '
                    'standard: standardizes features so that they are zero centered and '
                    'have unit variance. '
                    'minmax: scales the features based on minimum and maximum so that '
                    'they are between 0 and 1. '
                    'Default is unspecified which performs no scaling.')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=94,
                    help='The column number of the label in the input csv. Defautl is '
                    '94, set it otherwise')
parser.add_argument('-b', '--balanced_data', dest='balanced_data',
                    default=False, action='store_true',
                    help='Whether to balance the data we output. Default is false, '
                    'which means the number of men and women in the output could be '
                    'different. If true, data is balanced by undersampling majority '
                    'class and oversampling minority class')
args = parser.parse_args()

def main():
  df = pandas.read_csv(args.input_filename, index_col=False, header=0)
  data = df.values
  column_names = df.columns.values.tolist()
  
  # Extract features/labels and their names from raw data
  features = data[:, 0:args.label_column]
  labels = data[:, args.label_column].astype(int)
  feature_names = column_names[0:args.label_column]
  label_name = column_names[args.label_column]
  feature_label_names = feature_names + [label_name]

  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(features)
  features = imputer.transform(features)

  # scale data
  (features, dummy) = utils.scale_data(features, None, args.scaling_method)
  # stich data back together
  data = numpy.column_stack((features, labels))

  # Now balance the imputed data set.
  if args.balanced_data:
    features, labels = utils.balance_data(features, labels)
    # stich data back together
    data = numpy.column_stack((features, labels))

  # convert back to DF so that we can also write the column names
  df = pandas.DataFrame(data, columns=feature_label_names)
  df.to_csv(args.output_filename, index=False)

if __name__ == "__main__":
  main()
