#!/usr/bin/python

import argparse
import numpy
import sys
import csv
import utils
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer

parser = argparse.ArgumentParser(
    description='This script performs Principal Component Analysis on the original data')
parser.add_argument('-s', '--scaler', dest='scaling_method',
                    choices=['normal', 'standard', 'minmax', None], default='standard',
                    help = 'The type of scaling method to use prior to PCA. Possible '
                    'options are: normal, standard, minmax or None '
                    'normal: normalizes the features based on L2 norm. '
                    'standard: standardizes features so that they are zero centered and '
                    'have unit variance. '
                    'minmax: scales the features based on minimum and maximum so that '
                    'they are between 0 and 1. '
                    'None disables scaling. Default is standard.')
parser.add_argument('-c', '--num_components', dest='num_components',
                    default=None,
                    help='The number of components to keep. Possible values are plain '
                    'integers, "mle", or a number between 0 and 1 indicating the '
                    'fraction of variance that should be explained by the components. '
                    'Default is None which keeps all components.')
parser.add_argument('-lc', '--label_column', dest='label_column', type=int, default=94,
                    help='The column number of the label in the input csv. Defautl is '
                    '94, set it otherwise')
parser.add_argument('-o', '--transformed_output_filename',
                    dest='transformed_output_filename', default=None,
                    help='If provided, it determines the location of the ouptut filename '
                    'which will include the requested number of top components as '
                    'transformed features and the labels.')
parser.add_argument('-l', '--num_top_loadings', dest='num_top_loadings',
                    type=int, default=3,
                    help='Number of largest loadings (absolute value) per component to '
                    'print.')
parser.add_argument('input_filename')
parser.add_argument('loadings_output_filename')
parser.add_argument('explained_variance_output_filename')
args = parser.parse_args()

def main():
  # First read header columns
  input_file = open(args.input_filename, 'r')
  input_file_reader = csv.reader(input_file)
  headers = input_file_reader.next()
  input_file.close()

  # Let numpy know that NA corresponds to our missing value
  data=numpy.genfromtxt(args.input_filename, delimiter=",", skip_header=1,
                        missing_values="NA", filling_values = "NaN")
  # Impute the data and replace missing values
  imputer = Imputer(missing_values="NaN", strategy='mean', axis=0, copy=False)
  imputer.fit(data)
  data = imputer.transform(data)
  
  features = data[:, 0:args.label_column]
  labels = data[:, args.label_column:]
  
  # scale data
  (features, dummy) = utils.scale_data(features, None, args.scaling_method)

  num_components = args.num_components
  if num_components and num_components != 'mle':
    num_components = float(num_components)
    if num_components >= 1:
      num_components = int(num_components)
  
  pca = PCA(n_components = num_components, copy = True)
  transformed_features = pca.fit_transform(features)

  # write transformed features
  if args.transformed_output_filename:
    fields = []
    formats = []
    for i in range(1, num_components + 1):
      fields.append('component' + str(i))
      formats.append('%.20f')

    for i in range(1, data.shape[1] - args.label_column + 1):
      fields.append('label' + str(i))
      formats.append('%i')
    header = ','.join(fields)
    output_data = numpy.column_stack((transformed_features, labels))
    numpy.savetxt(args.transformed_output_filename, output_data,
                  comments = '', fmt = formats,
                  delimiter=',', header = header)
  
  # write component loading to output file
  loadings_output_file = open(args.loadings_output_filename, 'w')
  loadings_output_file_writer = csv.writer(loadings_output_file)
  loadings_output_file_writer.writerow(headers[0:args.label_column])
  for i in range(0, len(pca.components_)):
    component = pca.components_[i]
    loadings_output_file_writer.writerow(component)
  loadings_output_file.close()

  # Now write the individual and cumulative variance explained by each sucessive component
  explained_variance_output_file = open(args.explained_variance_output_filename, 'w')
  explained_variance_output_file_writer = csv.writer(explained_variance_output_file)
  explained_variance_output_file_writer.writerow(['Component_Number',
                                                  'Explained_Variance',
                                                  'Total_Explained_Variance'])
  total_explained_variance = 0
  for i in range(0, len(pca.components_)):
    explained_variance = pca.explained_variance_ratio_[i]*100.
    total_explained_variance += explained_variance
    explained_variance_output_file_writer.writerow([i+1, explained_variance,
                                                    total_explained_variance])
  explained_variance_output_file.close()

  # print top loadings per component
  for i in range(0, len(pca.components_)):
    print 'Top ' + str(args.num_top_loadings) + ' loadings for component ' + str(i)
    component = pca.components_[i]
    abs_component = map(abs, component)
    # Get the indices of components sorted by the features loading
    sorted_indices = [i[0] for i in sorted(enumerate(abs_component),
                                           key=lambda x: x[1],
                                           reverse=True)]
    for l in range(0, args.num_top_loadings):
      index = sorted_indices[l]
      print headers[index] + ' : ' + str(component[index])
  print '\n\n'
  print 'Explained variance ratio\n ' + str(pca.explained_variance_ratio_)
  print 'Total explained variance ' + str(numpy.sum(pca.explained_variance_ratio_))


if __name__ == "__main__":
  main()
