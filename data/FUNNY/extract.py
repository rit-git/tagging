import sys

root_directory = '../../'
sys.path.insert(0, root_directory + "pyfunctor")

import transform as transformer
import csv_handler as csv_handler
import json_handler as json_handler

review_input_path = "../FUNNY/yelp_academic_dataset_review.csv"
review_output_path = "./all.csv"

dataset = csv_handler.csv_readlines(review_input_path)

# get idx for review_id, text, and funny_count
def get_triplet_idx(header):
    idx_id = header.index('review_id')
    idx_text = header.index('text')
    idx_count = header.index('funny')
    return (idx_id, idx_text, idx_count)

def selector(row, idx_id, idx_text, idx_count):
    r_id = row[idx_id]
    text = row[idx_text]
    funny_count = int(row[idx_count])
    is_funny = None
    if (funny_count >= 5):
        is_funny = 1
    elif (funny_count == 0):
        is_funny = 0
    return (r_id, text, is_funny)

(idx_id, idx_text, idx_count) = get_triplet_idx(dataset[0])    

selected_datasets = transformer.map_func(dataset[1:], lambda line : selector(line, idx_id, idx_text, idx_count))

final_datasets = transformer.filter_func(selected_datasets, lambda row : row[2] != None)

csv_handler.csv_writelines(review_output_path, final_datasets)
