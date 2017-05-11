#!/usr/bin/python

import argparse
from pandas import read_csv

parser = argparse.ArgumentParser(
    description='Use this script to extract specific columns from the input csv and '
                'and write the column(s) to an output file.')
parser.add_argument('input_filename')
parser.add_argument('columns', help='Comma separated list of columns to extract.')
parser.add_argument('output_filename')
args = parser.parse_args()

def main():
  columns_to_extract = args.columns.split(',')

  df = read_csv(args.input_filename, sep=',', dtype=unicode)
  columns_df = df[columns_to_extract]
  columns_df.to_csv(args.output_filename, index=False)

if __name__ == "__main__":
  main()
