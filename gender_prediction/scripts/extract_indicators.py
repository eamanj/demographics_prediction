#!/usr/bin/python

import argparse
import csv
import sys
import bandicoot
import os
import tempfile

parser = argparse.ArgumentParser(
    description='Given a directory containing all CDRs, it computes bandicoot indicators '
                'for all users in the data.')
parser.add_argument('cdrs_input_dir',
                    help='The location of CDRs. This directory will have many '
                    'subdirectories and each subdirectory contains the CDRs for several '
                   'users, one file per user. So the structure is directories of CDR '
                    'files nested in this input directory.')
parser.add_argument('antennas_input_file',
                    help='The location of antennas input file. It contains a mapping '
                    'from antenna_id to its latitude and longitude. The header should '
                    'look like: place_id,latitude,longitude');
parser.add_argument('indicators_output',
                    help='The location where output indicators will be written to. One '
                    'user per line.')
parser.add_argument('-d', '--delete_bad_columns', dest='delete_bad_columns',
                    default=False, action='store_true',
                    help = 'Bandicoot outpus columns that are always None for every '
                    'user. If requested, these columns will be deleted from the output '
                    'csv.')
parser.add_argument('-a', '--attributes_input_file', dest='attributes_input_file',
                    help = 'A csv file containing all user attributes, one user per '
                    'line. If specified, gender and age attributes will also be added to '
                    'the final indicators csv file, as seperate columns. The header must '
                    'exist and contain the following strings: "phone_hash", "age" and '
                    '"gender". Their order does not matter as long as their position in '
                    'the header matches their position in each line.')
args = parser.parse_args()

# returns a mapping from phone_hash to (gender,age)
def read_user_attributes(attributes_input_file):
  attributes_file = open(attributes_input_file, "r")
  attributes_file_reader = csv.reader(attributes_file, delimiter="|")
  keys = attributes_file_reader.next()
  attribute_keys_idx = dict()
  # find index of attributes from the header
  for i in range(len(keys)):
    if keys[i] in ["gender", "age", "phone_hash"]:
      attribute_keys_idx[keys[i]] = i

  if len(attribute_keys_idx) != 3:
    sys.exit('Only found ' + str(attribute_keys_idx) +
             ' attributes in attributes file: ' + attributes_input_file)
  
  # read in the attributes now that we have their indices
  user_attributes = dict()
  for row in attributes_file_reader:
    phone_hash = row[attribute_keys_idx["phone_hash"]]
    gender = row[attribute_keys_idx["gender"]]
    if gender == 'MASCULINO':
      gender = 1
    elif gender == 'FEMENINO':
      gender = 0
    else:
      sys.exit('Bad Gender: ' + gender + ' in attributes')

    age = row[attribute_keys_idx["age"]]
    user_attributes[phone_hash] = (gender, age)
  
  attributes_file.close()
  return user_attributes


# deletes some columns that are always none in bandicoot output
def delete_bad_columns(indicators_output):
  # Supress columns that are always None
  bad_keys = ['response_delay_text__callandtext',
              'radius_of_gyration',
              'number_of_contacts__text',
              'number_of_interactions__text',
              'percent_nocturnal__text',
              'entropy_of_contacts__text',
              'interactions_per_contact__text',
              'interevents_time__text',
              'number_of_contacts__call',
              'number_of_interactions__call',
              'duration_of_calls__call',
              'percent_nocturnal__call',
              'percent_initiated_interactions__call',
              'entropy_of_contacts__call',
              'interactions_per_contact__call',
              'interevents_time__call',
              'percent_at_home']
  tmp_indicators_output_file =  indicators_output + "_tmp"
  os.rename(indicators_output, tmp_indicators_output_file)
  tmp_file = open(tmp_indicators_output_file, "r")
  tmp_file_reader = csv.reader(tmp_file)
  keys = tmp_file_reader.next()
  tmp_file.close()
  good_keys_idx = list()
  for i in range(len(keys)):
    if keys[i] not in bad_keys:
      good_keys_idx.append(i)

  # reopen and rewrite now that we have good column indices 
  tmp_file = open(tmp_indicators_output_file, "r")
  tmp_file_reader = csv.reader(tmp_file)
  output_file = open(indicators_output, "w")
  output_file_writer = csv.writer(output_file)
  for row in tmp_file_reader:
    output_file_writer.writerow([row[i] for i in good_keys_idx])
  output_file.close()
  tmp_file.close()
  # remove tmp file now that we have it without bad columns
  os.remove(tmp_indicators_output_file)


def main():
  # read in attributes for all user
  if args.attributes_input_file:
    user_attributes = read_user_attributes(args.attributes_input_file)

  indicators = []
  for dir in os.listdir(args.cdrs_input_dir):
    cdrs_dir = os.path.join(args.cdrs_input_dir, dir)

    if not os.path.isdir(cdrs_dir):
      continue
    for file in os.listdir(cdrs_dir):
      cdr_input_file = os.path.join(cdrs_dir, file)
      user_name = file
      
      # should we add atributes to indicators?
      if args.attributes_input_file:
        # find the attributes for this user and write them to an in-memory file
        if user_name not in user_attributes:
          sys.exit('Could not find attributes for user ' + user_name)
        gender = user_attributes[user_name][0]
        age = user_attributes[user_name][1] if user_attributes[user_name][1] else "None"
        attributes_file = tempfile.NamedTemporaryFile()
        attributes_file.write("key,value\n")
        attributes_file.write("gender," + str(gender) + "\n")
        attributes_file.write("age," + str(age))
        # don't forget to seek to beginning so that what you wrote can be read!
        attributes_file.seek(0)
        user = bandicoot.read_csv(cdr_input_file, args.antennas_input_file,
                                  attributes_path=attributes_file.name,
                                  describe=False)
        attributes_file.close()
      else:
        user = bandicoot.read_csv(cdr_input_file, args.antennas_input_file, describe=False)
      
      # set the username to filename
      user.name = user_name
    
      # Extended summary also computes skewness and kurtosis
      user_indicators = bandicoot.utils.all(user, weekly=True, summary='extended',
                                            attributes=True)

      indicators.append(user_indicators)

  bandicoot.io.to_csv(indicators, args.indicators_output)

  if args.delete_bad_columns:
    delete_bad_columns(args.indicators_output)


if __name__ == "__main__":
  main()
