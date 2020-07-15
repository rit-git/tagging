from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import sys
import os
sys.path.insert(0,  os.environ['TAG_HOME'] + "/pyfunctor")
import csv_handler as csv_handler
import transform as transform

def log_to_csv(y_true, y_pred, csv_log_file_path, identity_info="dataset"):
    labels = [0, 1]
    result = precision_recall_fscore_support(y_true, y_pred)

    row = []
    row.append(identity_info)

    # neg 
    row.append('label ' + str(labels[0]) + ":")
    row.append(result[0][0])
    row.append(result[1][0])
    row.append(result[2][0])
    row.append(result[3][0])

    row.append(' ')
    
    # pos
    row.append('label ' + str(labels[1]) + ":")
    row.append(result[0][1])
    row.append(result[1][1])
    row.append(result[2][1])
    row.append(result[3][1])

    csv_handler.append_row(csv_log_file_path, row)

def log_to_csv_multi_f1(y_true, y_pred, csv_log_file_path, identity_info="dataset"):
    result = precision_recall_fscore_support(y_true, y_pred)

    row = []
    row.append(identity_info)

    multi_f1 = list(result[2])
    row += multi_f1

    csv_handler.append_row(csv_log_file_path, row)

def log_to_csv_with_auc_accuracy(y_true, y_pred, y_score, csv_log_file_path, identity_info="dataset"):
    labels = [0, 1]
    result = precision_recall_fscore_support(y_true, y_pred)

    row = []
    row.append(identity_info)

    # neg
    row.append('label ' + str(labels[0]) + ":")
    row.append(result[0][0])
    row.append(result[1][0])
    row.append(result[2][0])
    row.append(result[3][0])

    row.append(' ')
    
    # pos
    row.append('label ' + str(labels[1]) + ":")
    row.append(result[0][1])
    row.append(result[1][1])
    row.append(result[2][1])
    row.append(result[3][1])

    row.append(' ')

    # auc and accuracy
    y_pos_score = transform.map_func(y_score, lambda p : p[1])
    auc = metrics.roc_auc_score(y_true, y_pos_score)
    row.append(auc)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    row.append(accuracy)

    csv_handler.append_row(csv_log_file_path, row)
