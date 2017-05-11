#!/usr/bin/python

import argparse
import csv
import sys
import os  

parser = argparse.ArgumentParser(
    description='Given a directory containing all CDRs, it extracts all antennas '
                'and outputs a csv with the three columns: antenna_id, latitude, '
                'longitude. Each antenna will receive a unique id.')
parser.add_argument('cdrs_dir_input')
parser.add_argument('antenna_output')
args = parser.parse_args()

def main():
  if not os.path.isdir(args.cdrs_dir_input) or not os.path.exists(args.cdrs_dir_input):
    sys.exit("The directory " + args.cdrs_dir_input + " does not exist.")

  # a dictionary from (lat,lng) to antenna-id
  antennas = dict()
  # the initial antenna id
  antenna_id = 1
  for input_file in os.listdir(args.cdrs_dir_input):
    cdrs_input_file = os.path.join(args.cdrs_dir_input, input_file)
    if os.path.isfile(cdrs_input_file):
      with open(cdrs_input_file, 'r') as f:
        line_num = 0
        cdrs_input_reader = csv.reader(f, delimiter = '\t')
        for row in cdrs_input_reader:
          line_num += 1
          row = [x.strip() for x in row]
          if len(row) < 4:
            sys.exit('Bad CDR row: ' + str(row) + ' in line ' + str(line_num))
            
          # phone hash of customer is in second fields of both sms and call records
          latitude = row[7]
          longitude = row[8]
          if not latitude or not longitude:
            # empty antenna location, skip
            continue

          key = (latitude, longitude)
          if not key in antennas:
            antennas[key] = antenna_id
            antenna_id += 1

  with open(args.antenna_output, 'w') as f:
    antenna_output_writer = csv.writer(f, delimiter=',')
    # write header
    antenna_output_writer.writerow(["place_id", "latitude", "longitude"])
    for antenna_location, antenna_id in antennas.iteritems():
      latitude = antenna_location[0]
      longitude = antenna_location[1]
      antenna_output_writer.writerow([antenna_id, latitude, longitude])

if __name__ == "__main__":
  main()


