#!/usr/bin/python

import argparse
from collections import defaultdict
from operator import add
import pandas
import sys

parser = argparse.ArgumentParser(description='Script replaces "None" strings or missing '
    'data in raw CDR data, removes some bad rows according to some filters such as '
    'threshold on min active days and min number of contacts, and writes the cleaned '
    'version to output file. The cleaned version is a subset of the input file in both '
    'rows and columns, with the exception that "None" string or missing data is '
    'replaced.')
parser.add_argument('input_filename')
parser.add_argument('output_filename')
parser.add_argument('-n', '--na_replacement', dest='na_replacement',
                    help = "Value specified as None will be replaced with this string",
                    default="NA")
parser.add_argument('-a', '--active_days', dest='min_active_days',
                    help = 'Users whose active_days__callandtext__mean is less than this '
                    'threshold will be dropped', default=2)
parser.add_argument('-c', '--num_contacts', dest='min_num_contacts',
                    help = 'Users whose number_of_contacts__call__mean is less than this '
                    'threshold will be dropped', default=2)
parser.add_argument('-m', '--missing_locations', dest='max_missing_locations',
                    help = 'Users whose records_missing_location is more than this '
                    'threshold will be dropped', default=1000000)
parser.add_argument('-hc', '--valid_has_call', dest='valid_has_call',
                    default=False, action='store_true',
                    help='Whether to filter out rows whose has_call is False. In effect '
                    'only users with call activity will be kept. Default is False.')
parser.add_argument('-ht', '--valid_has_text', dest='valid_has_text',
                    default=False, action='store_true',
                    help='Whether to filter out rows whose has_text is False. In effect '
                    'only users with text activity will be kept. Default is False.')
parser.add_argument('-rg', '--valid_radius_gyration', dest='valid_radius_gyration',
                    default=False, action='store_true',
                    help='Whether to filter out rows whose radius of gyration field '
                    'is invalid: None. Default is false which keeps records which do '
                    'not have valid radius of gyration.')
parser.add_argument('-cr', '--columns_to_remove', dest='columns_to_remove',
                    default='',
                    help='If provided, it should be name of a file containing the '
                    'columns to remove, one column per line. If not provided, no columns '
                    'will be removed.')
args = parser.parse_args()

def data_missing(data):
  missing_values = ['None', 'nan', '']
  if (any(data == missing_value for missing_value in missing_values) or
      pandas.isnull(data)):
    return True
  return False

def row_has_missing_data(row, indicators):
  for indicator in indicators:
    if data_missing(row[indicator]):
      return True
  return False

# The following methods, return the corresponding columne throughout the whole week based
# on the version of data row
def row_number_of_contacts_call(row, version):
  if version < '0.3':
    return row['number_of_contacts__call__mean']
  else:
    return row['number_of_contacts__allweek__allday__call__mean']
    
def row_active_days(row, version):
  if version < '0.3':
    return row['active_days__callandtext__mean']
  else:
    return row['active_days__allweek__allday__callandtext__mean']
    
def row_has_call(row, version):
  if version < '0.3':
    return True if row['has_call'] == 'True' else False
  else:
    return True if row['reporting__has_call'] == 'True' else False

def row_has_text(row, version):
  if version < '0.3':
    return True if row['has_text'] == 'True' else False
  else:
    return True if row['reporting__has_text'] == 'True' else False

def row_radius_of_gyration(row, version):
  if version < '0.3':
    return row['radius_of_gyration__mean']
  else:
    return row['radius_of_gyration__allweek__allday__mean']

def row_records_missing_locations(row, version):
  if version < '0.3':
    return row['records_missing_locations']
  else:
    missing = (float(row['reporting__number_of_records']) *
               float(row['reporting__percent_records_missing_location']))
    return int(missing)

def main():
  na_replacement = args.na_replacement
  min_active_days =  float(args.min_active_days)
  min_num_contacts =  float(args.min_num_contacts)
  max_missing_locations = int(args.max_missing_locations)

  df = pandas.read_csv(args.input_filename, sep=',', dtype=unicode)

  # first remove unnecessary columns
  columns_to_remove = set()
  if args.columns_to_remove:
    with open(args.columns_to_remove) as f:
      columns_to_remove.update([x.strip() for x in f.readlines()])

  # Manually add attributes at the end of indicators so that labels are last two
  # columns. But the income should be the first of the attributes
  indicators = list()
  attributes = list()
  for col in df.columns:
    if col in columns_to_remove:
      continue
    elif "attributes" in col:
      if "income" in col:
        income_column_name = col
      else:
        attributes.append(col)
    else:
      indicators.append(col)
  # sort column names so that repeated runs of this program generates the same output
  indicators.sort()
  indicators.append(income_column_name)
  indicators.extend(attributes)

  output_file = open(args.output_filename, 'w')
  # write header
  output_file.write(','.join(indicators) + "\n")
  num_rejected_lines = 0
  num_accepted_lines = 0
  
  num_rejected_lines_with_missing_data = 0
  num_accepted_lines_with_missing_data = 0
  
  # how genders are represented in (south asian) data
  FEMALE = "2"
  MALE = "1"

  version_column = [s for s in df.columns if 'version' in s]
  if len(version_column) == 0 or len(version_column) > 1:
    sys.exit('Bad version column(s). columns matching version: ' + str(version_column))
  version_column = version_column[0]

  for index, row in df.iterrows():
    version = row[version_column]
    reject_line = False
    # filter rows that have missing number_of_contacts__call__mean or whose number of
    # contacts is less than a threshold
    num_contacts = row_number_of_contacts_call(row, version)
    if data_missing(num_contacts) or float(num_contacts) < min_num_contacts:
      reject_line = True

    # filter rows that have missing active_days__callandtext__mean or whose number of
    # active days is less than a threshold
    active_days = row_active_days(row, version)
    if data_missing(active_days) or float(active_days) < min_active_days:
      reject_line = True

    # filter out rows that do not have call data
    has_call = row_has_call(row, version)
    if args.valid_has_call and not has_call:
      reject_line = True
    
    # filter out rows that do not have text data
    has_text = row_has_text(row, version)
    if args.valid_has_text and not has_text:
      reject_line = True

    radius_of_gyration = row_radius_of_gyration(row, version)
    if args.valid_radius_gyration and data_missing(radius_of_gyration):
      reject_line = True

    records_missing_locations = row_records_missing_locations(row, version)
    if int(records_missing_locations) > max_missing_locations:
      reject_line = True

    has_missing_data = row_has_missing_data(row, indicators)
    if reject_line:
      num_rejected_lines += 1
      if has_missing_data:
        num_rejected_lines_with_missing_data += 1
      continue

    features = list()
    for indicator in indicators:
      feature = row[indicator] if not data_missing(row[indicator]) else na_replacement
      # fix gender to only zero/one
      if "gender" in indicator:
        feature = "0" if feature == FEMALE else MALE
      features.append(feature)

    if has_missing_data:
      num_accepted_lines_with_missing_data += 1
    num_accepted_lines += 1

    feature_string = ','.join(str(x) for x in features)
    output_file.write(feature_string + "\n")
 
  print('Accepted ' + str(num_accepted_lines) + ' lines')
  print('Rejected ' + str(num_rejected_lines) + ' lines based on the thresholds.')
  print(str(num_accepted_lines_with_missing_data) +
      ' accepted lines have missing data in them.')
  print(str(num_rejected_lines_with_missing_data) +
      ' rejected lines have missing data in them.')
  output_file.close()


if __name__ == "__main__":
  main()
