import sys
sys.path.insert(0, "../../pyfunctor")
import transform as transform
import csv_handler as csv_handler
from sklearn.metrics import precision_recall_fscore_support
import math
import os

def get_threds_by_min_max_even(y_pred_score, num_threds):
    max_score = max(y_pred_score)
    min_score = min(y_pred_score)

    thred = max_score
    step_size = (max_score - min_score) / num_threds


    thred_col = []
    while thred > min_score:
        thred_col.append(thred)
        thred -= step_size

    thred_col.append(min_score)

    return thred_col

def get_threds_by_sorted_score_equal_length(y_pred_score, num_threds):
    sorted_pred_score = sorted(y_pred_score, key=lambda score : -score)
    step = int(len(sorted_pred_score) / num_threds)
    thred_col = sorted_pred_score[0::step]

    return thred_col

if __name__ == "__main__":
    csv_input_path = sys.argv[1]
    y_true_col = int(sys.argv[2])
    y_pred_col = int(sys.argv[3])
    num_threds = int(sys.argv[4])
    csv_output_path = sys.argv[5]

    print_header = '1'
    if len(sys.argv) > 6:
        print_header = sys.argv[6]
    assert(print_header == '1' or print_header == '0')

    thred_method = "min_max_even"
    if os.path.exists(csv_output_path):
        os.remove(csv_output_path)

    csv_dataset = csv_handler.csv_readlines(csv_input_path)

    y_true = transform.map_func(csv_dataset, lambda row : int(row[y_true_col]))
    y_pred_score = transform.map_func(csv_dataset, lambda row : float(row[y_pred_col]))


    #y_pred_score = transform.map_func(y_pred_score, lambda score : 1 / (1 + math.exp(-score)))

    thred_col = []

    if thred_method == "min_max_even":
        thred_col = get_threds_by_min_max_even(y_pred_score, num_threds)

    else: # sorted score even slot
        thred_col = get_threds_by_sorted_score_equal_length(y_pred_score, num_threds)

    if print_header == '1':
        csv_handler.append_row(csv_output_path, ['threshold', 'precision', 'recall', 'fscore', 'support'])

    for thred in thred_col:
        y_pred = transform.map_func(y_pred_score, lambda score: 1 if score >= thred else 0)
        result = precision_recall_fscore_support(y_true, y_pred)
        
        csv_handler.append_row(csv_output_path, [thred, result[0][1], result[1][1], result[2][1], result[3][1]])
