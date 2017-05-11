#!/usr/bin/python

import argparse
import csv
from collections import defaultdict

parser = argparse.ArgumentParser(
  description='Simple script for extracting the results for plotting from raw outputs '
              'of scripts.')
parser.add_argument('linear_svm_input_filename')
parser.add_argument('kernel_svm_input_filename')
parser.add_argument('random_forest_input_filename')
parser.add_argument('logistic_input_filename')
parser.add_argument('knn_input_filename')
parser.add_argument('full_accuracy_output_filename')
parser.add_argument('stats_per_gender_output_filename')
args = parser.parse_args()

# Line numbers where the results start
LINEAR_SVM_RESULTS_LINE = 19
KERNEL_SVM_RESULTS_LINE = 19
RANDOM_FOREST_RESULTS_LINE = 41
LOGISTIC_RESULTS_LINE = 19
KNN_RESULTS_LINE = 19


def write_full_accuracy_results(linear_svm_accuracies,
                                kernel_svm_accuracies,
                                random_forest_accuracies,
                                logistic_accuracies,
                                knn_accuracies,
                                output_filename):
  header = ["Percentage-predicted",
            "SVM-L","SVM-K","RF","LOGISTIC","KNN"]
  output_file = open(output_filename, 'w')
  output_file_writer = csv.writer(output_file)
  output_file_writer.writerow(header)

  for percentage_predicted in range(5, 101):
    linear_svm_accuracy = linear_svm_accuracies[percentage_predicted]
    kernel_svm_accuracy = kernel_svm_accuracies[percentage_predicted]
    random_forest_accuracy = random_forest_accuracies[percentage_predicted]
    logistic_accuracy = logistic_accuracies[percentage_predicted]
    knn_accuracy = knn_accuracies[percentage_predicted]
    
    row = [percentage_predicted,
           linear_svm_accuracy, kernel_svm_accuracy, random_forest_accuracy,
           logistic_accuracy, knn_accuracy]
    output_file_writer.writerow(row)

  output_file.close()


def extract_accuracy_per_percentage_predicted(input_filename, starting_line_num):
  input_file = open(input_filename, 'r')
  input_file_reader = csv.reader(input_file)

  # current location of reader
  current_line_num = 1
  while current_line_num < starting_line_num:
    input_file_reader.next()
    current_line_num += 1

  accuracy_per_percentage = defaultdict(lambda: '')
  for row in input_file_reader:
    accuracy = row[1]
    percentage_predicted = int(round(float(row[2])))
    accuracy_per_percentage[percentage_predicted] = accuracy

  input_file.close()
  return accuracy_per_percentage 

def extract_stats_per_gender(input_filename, starting_line_num):
  input_file = open(input_filename, 'r')
  input_file_reader = csv.reader(input_file)

  # current location of reader
  current_line_num = 1
  while current_line_num < starting_line_num:
    input_file_reader.next()
    current_line_num += 1

  # figure out number of men and women
  num_total_males = 0
  num_total_females = 0
  for row in input_file_reader:
    true_females = int(row[3])
    false_males = int(row[4])
    true_males = int(row[7])
    false_females = int(row[8])
    
    num_total_females = max(num_total_females, true_females + false_males)
    num_total_males = max(num_total_males, true_males + false_females)

  # rewind the file
  input_file.seek(0)
  current_line_num = 1
  while current_line_num < starting_line_num:
    input_file_reader.next()
    current_line_num += 1
  
  # now extract info
  male_stats = list()
  female_stats = list()
  for row in input_file_reader:
    true_females = int(row[3])
    false_males = int(row[4])
    true_males = int(row[7])
    false_females = int(row[8])
    
    true_female_num_predicted = true_females + false_males
    female_num_predicted = true_females + false_females
    female_percent_predicted = round(true_female_num_predicted*100.0 / num_total_females, 3)
    female_recall = (round(true_females*100.0 / true_female_num_predicted, 3) if
                     true_female_num_predicted else 0)
    female_precision = (round(true_females*100.0 / female_num_predicted, 3) if
                        female_num_predicted else 0)
    female_stats.append((true_female_num_predicted, female_percent_predicted,
                         true_females, false_females,
                         female_recall, female_precision))

    true_male_num_predicted = true_males + false_females
    male_num_predicted = true_males + false_males
    male_percent_predicted = round(true_male_num_predicted*100.0 / num_total_males, 3)
    male_recall = (round(true_males*100.0 / true_male_num_predicted, 3) if
                   true_male_num_predicted else 0)
    male_precision = (round(true_males*100.0 / male_num_predicted, 3) if
                      male_num_predicted else 0)
    male_stats.append((true_male_num_predicted, male_percent_predicted,
                       true_males, false_males,
                       male_recall, male_precision))
  input_file.close()

  # sort by num_predicted
  male_stats_sorted = sorted(male_stats, key=lambda x: x[0])
  female_stats_sorted = sorted(female_stats, key=lambda x: x[0])
  return (male_stats_sorted, female_stats_sorted)


def write_stats_per_gender_results(linear_svm_stats,
                                   kernel_svm_stats,
                                   random_forest_stats,
                                   logistic_stats,
                                   knn_stats,
                                   output_filename):

  header = []
  methods = ["SVM-L", "SVM-K", "RF", "LOGISTIC", "KNN"]
  genders = ["Male", "Female"]
  for gender in genders:
    for method in methods:
      prefix = gender + "-" + method
      header.extend([prefix + "-num-predicted",
                     prefix + "-percent-predicted",
                     prefix + "-true",
                     prefix + "-false",
                     prefix + "-Recall",
                     prefix + "-Precision"])
  
  output_file = open(output_filename, 'w')
  output_file_writer = csv.writer(output_file)
  output_file_writer.writerow(header)

  data = defaultdict(lambda: defaultdict(list))
  data["Male"]["SVM-L"] = linear_svm_stats[0]
  data["Male"]["SVM-K"] = kernel_svm_stats[0]
  data["Male"]["RF"] = random_forest_stats[0]
  data["Male"]["LOGISTIC"] = logistic_stats[0]
  data["Male"]["KNN"] = knn_stats[0]
  data["Female"]["SVM-L"] = linear_svm_stats[1]
  data["Female"]["SVM-K"] = kernel_svm_stats[1]
  data["Female"]["RF"] = random_forest_stats[1]
  data["Female"]["LOGISTIC"] = logistic_stats[1]
  data["Female"]["KNN"] = knn_stats[1]
  
  num_rows = max(len(linear_svm_stats[0]),
                 len(kernel_svm_stats[0]),
                 len(random_forest_stats[0]))

  for i in range(0, num_rows):
    row = []
    for gender in genders:
      for method in methods:
        row.extend(data[gender][method][i])

    output_file_writer.writerow(row)
  
  output_file.close()


def main():
  # full accuracy
  linear_svm_accuracies = extract_accuracy_per_percentage_predicted(
      args.linear_svm_input_filename, LINEAR_SVM_RESULTS_LINE)
  kernel_svm_accuracies = extract_accuracy_per_percentage_predicted(
      args.kernel_svm_input_filename, KERNEL_SVM_RESULTS_LINE)
  random_forest_accuracies = extract_accuracy_per_percentage_predicted(
      args.random_forest_input_filename, RANDOM_FOREST_RESULTS_LINE)
  logistic_accuracies = extract_accuracy_per_percentage_predicted(
      args.logistic_input_filename, LOGISTIC_RESULTS_LINE)
  knn_accuracies = extract_accuracy_per_percentage_predicted(
      args.knn_input_filename, KNN_RESULTS_LINE)

  write_full_accuracy_results(linear_svm_accuracies,
                              kernel_svm_accuracies,
                              random_forest_accuracies,
                              logistic_accuracies,
                              knn_accuracies,
                              args.full_accuracy_output_filename)
 
  # stats per gender
  linear_svm_stats = extract_stats_per_gender(args.linear_svm_input_filename,
                                              LINEAR_SVM_RESULTS_LINE)
  kernel_svm_stats = extract_stats_per_gender(args.kernel_svm_input_filename,
                                              KERNEL_SVM_RESULTS_LINE)
  random_forest_stats = extract_stats_per_gender(args.random_forest_input_filename,
                                                 RANDOM_FOREST_RESULTS_LINE)
  logistic_stats = extract_stats_per_gender(args.logistic_input_filename,
                                            LOGISTIC_RESULTS_LINE)
  knn_stats = extract_stats_per_gender(args.knn_input_filename,
                                       KNN_RESULTS_LINE)
  
  write_stats_per_gender_results(linear_svm_stats,
                                 kernel_svm_stats,
                                 random_forest_stats,
                                 logistic_stats,
                                 knn_stats,
                                 args.stats_per_gender_output_filename)

if __name__ == "__main__":
  main()
