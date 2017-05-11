#!/usr/bin/python

import os
import sys
import argparse
import csv

parser = argparse.ArgumentParser(
  description='Simple script for joining indicators with atributes in two different '
              'files. Joining is done based on hash column. The output file will contain '
              'all indicators plus the attributes.')
parser.add_argument('indicators_input_filename')
parser.add_argument('attributes_input_filename')
parser.add_argument('output_filename')
parser.add_argument('-ac', '--attribute_columns', dest='attribute_columns',
                    default="1,2",
                    help='The comma-seperated column numbers of the atribute to join in '
                    'the attributes input file. The default is 1,2 and starts at zero.')
parser.add_argument('-ijc', '--indicators_join_column', dest='indicators_join_column',
                    type=int, default=0,
                    help='The column number of the joining id in the indicators input '
                    'file. The default is 0.')
parser.add_argument('-ajc', '--attributes_join_column', dest='attributes_join_column',
                    type=int, default=0,
                    help='The column number of the joining id in the attributes input '
                    'file. The default is 0.')
args = parser.parse_args()

if __name__ == "__main__":
  id_hash_to_attributes = dict()
  if not os.path.isfile(args.attributes_input_filename):
    sys.exit('attributes_input file : ' + args.attributes_input_filename +
             ' is not a file.')

  attribute_file = open(args.attributes_input_filename, 'r')
  csvreader = csv.reader(attribute_file)
  
  # get attribute names from header
  attribute_indices = [int(i) for  i in args.attribute_columns.split(',')]
  header = csvreader.next()
  attribute_names = [header[i] for i in attribute_indices]

  for line in csvreader:
    id = line[args.attributes_join_column]
    attributes = [line[i] for i in attribute_indices]
    id_hash_to_attributes[id] = attributes
  attribute_file.close()

  output_file = open(args.output_filename, 'w')
  csvwriter = csv.writer(output_file, delimiter=',')
  indicators_file = open(args.indicators_input_filename, 'r')
  indicators_csvreader = csv.reader(indicators_file)

  # write the header
  output_headers = indicators_csvreader.next()
  output_headers.extend(attribute_names)
  csvwriter.writerow(output_headers)

  rows_missing_attributes = list()
  for row in indicators_csvreader:
    id_hash = row[args.indicators_join_column]
    if id_hash not in id_hash_to_attributes:
      rows_missing_attributes.append(id_hash)
    else:
      attributes = id_hash_to_attributes[id_hash]
      output_row = row
      output_row.extend(attributes)
      csvwriter.writerow(output_row)

  output_file.close()
  print ('There were ' + str(len(rows_missing_attributes)) +
         ' rows with missing attributes.')
  if len(rows_missing_attributes):
    print ('Some rows with missing attributes:')
    for i in range(0,10):
      print str(rows_missing_attributes[i])
