#!/usr/bin/python

import argparse
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score

parser = argparse.ArgumentParser(description='plot confusion matrix.')
parser.add_argument('input_filename')
parser.add_argument('output_figure')
args = parser.parse_args()
 
def plot_confusion_matrix(cm, class_names, title='Confusion matrix', cmap=plt.cm.Blues):
  font_size = 14
  plt.imshow(cm, interpolation='nearest', cmap=cmap, origin='lower')
  plt.title(title)
  cb = plt.colorbar()
  cb.ax.tick_params(labelsize=font_size)
  tick_marks = numpy.arange(len(class_names))
  plt.xticks(tick_marks, class_names, fontsize = font_size)
  plt.yticks(tick_marks, class_names, fontsize = font_size)
  plt.ylabel('True', fontsize = font_size)
  plt.xlabel('Predicted', fontsize = font_size)
  plt.tight_layout()
  
def annotate_plot(cm):
  # note that column position in matrix (j) corresponds to x-coordinate
  for i in range(0, cm.shape[0]):
    for j in range(0, cm.shape[1]):
      plt.annotate(round(cm[i,j], 2),
                   xy=(j, i), xytext=(j, i), textcoords='data',
                   fontsize = 13,
                   horizontalalignment='center', verticalalignment='center')

def main():
  cm = numpy.loadtxt(args.input_filename, dtype=int, delimiter=',')
  numpy.set_printoptions(precision=2)
  
  print('Confusion Matrix Without Normalization')
  print str(cm)
  
  print('Confusion Matrix With Normalization')
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
  print(cm_normalized)

  class_values = [1,2,3,4,5]
  class_names = map(str, class_values)
  first_class = class_values[0]

  y_true = list()
  y_pred = list()
  num_rows = cm.shape[0]
  num_columns = cm.shape[1]
  for row  in range(num_rows):
    for column in range(num_columns):
      y_true.extend([first_class + row]*cm[row,column])
      y_pred.extend([first_class + column]*cm[row,column])

  # Predict test and report full stats
  print("\n*****************************\n")
  print('MAE: {}'.format(
      mean_absolute_error(y_true, y_pred, multioutput='uniform_average')))
  
  print('Test Accuracy: {}'.format(accuracy_score(y_true, y_pred)*100.))
  print('Classification report:')
  print(classification_report(y_true, y_pred, class_values))
  print('Weighted Precision Recall:')
  print(precision_recall_fscore_support(y_true, y_pred, labels=class_values,
                                        pos_label=None,
                                        average='weighted'))
  print('Unweighted Precision Recall:')
  print(precision_recall_fscore_support(y_true, y_pred, labels=class_values,
                                        pos_label=None,
                                        average='macro'))
  # print and plot confusion matrix
  plt.figure(1)
  plot_confusion_matrix(cm, class_names, '')
  annotate_plot(cm_normalized)
  plt.savefig(args.output_figure + '_unnormalized.pdf', format='pdf')
  plt.close()


  # Normalize the confusion matrix by row (i.e by the number of samples
  # in each class)
  plt.figure(2)
  plot_confusion_matrix(cm_normalized, class_names, '')
  plt.savefig(args.output_figure + '_normalized.pdf', format='pdf')

if __name__ == '__main__':
  main()
