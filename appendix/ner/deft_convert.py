import sys
from os import listdir
sys.path.insert(0, "../../pyfunctor")
import csv_handler as csv_handler
import transform as transform


dataset = "dev"
input_dir = "./deft_corpus/data/deft_files/" + dataset
output_path = dataset + ".csv"

label_map = {"B" : 0, "I" : 1, "O" : 2}

def distill(input_path, label_map):
    dataset = csv_handler.csv_readlines(input_path, delimit='\t')
    dataset = transform.filter_func(dataset, lambda row : len(row) >= 5)
    token_label = transform.map_func(dataset, lambda row : [row[0].strip(), label_map[row[4].strip()[0]]])
    return token_label

final = []
for file_name in listdir(input_dir):
    token_label = distill(input_dir + "/" + file_name, label_map)
    final += token_label

final = transform.indexleft_flat_func(final)
csv_handler.csv_writelines(output_path, final)
