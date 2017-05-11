#!/usr/bin/python

import argparse
from collections import defaultdict
from operator import add
from pandas import read_csv

# input should be already cleaned by clean_data.py
# it should have all Nones replaced by NA

parser = argparse.ArgumentParser(description='Script takes already cleaned european or '
    'south asian data (output of clean_data.py) and transforms it into a form usable by '
    'libsvm. This script replaces NA values with the mean of the corresponding column.')
parser.add_argument('input_filename')
parser.add_argument('output_filename')
parser.add_argument('-s', '--south_asian', dest='south_asian_data', default=False,
                    action='store_true',
                    help="Whether to process south asian data. Default is false, which means the input "
                    "is cleaned european data.")
args = parser.parse_args()

def GetFeaturesString(features):
    features_string = ''
    for (feature_id, feature) in enumerate(features):
        features_string += "%d:%.5f " % (feature_id + 1, feature)
    
    return features_string

def main():
  df = read_csv(args.input_filename)

  # now find the average of each row, so we can fill missing columns
  # we are assuming the input file has replaced Nones with NA here.
  means = df.mean()
  df.fillna(df.mean(), inplace=True)

  # Set the correct name for label columns based on the source of data.
  if args.south_asian_data:
    gender_column = 'attributes__surv_gender'
    age_column = 'attributes__surv_act_age'
  else:
    gender_column = 'attributes__survey_gender'
    age_column = 'attributes__survey_age'

  indicators = [col for col in df.columns if col not in [gender_column, age_column]]
  MALE = "+1"
  FEMALE = "-1"
  output_file = open(args.output_filename, 'w')
  num_rejected_lines = 0
  num_accepted_lines = 0
  for index, row in df.iterrows():
    features = list()
    for indicator in indicators:
      features.append(row[indicator])

    num_accepted_lines += 1

    gender = FEMALE
    if row[gender_column] == 1:
      gender = MALE

    feature_string = GetFeaturesString(features)

    output_line = (gender + ' ' + feature_string + "\n")
    output_file.write(output_line)
 
  output_file.close()


if __name__ == "__main__":
  main()
