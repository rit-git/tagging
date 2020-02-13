import sys
sys.path.insert(0, "../../pyfunctor")

import transform as transformer
import csv_handler as csv_handler

data_dir = "./dataset/"
train_id_path = data_dir + "train_ids.txt"
test_id_path = data_dir + "test_ids.txt"
anno_dir= data_dir + '/iclr_anno_final/'
train_output_path = data_dir + "./trainset.csv"
test_output_path = data_dir + "./devset.csv"

import csv_handler as csv_handler
train_id_dataset = open(train_id_path, "r").read().splitlines()

test_id_dataset =  open(test_id_path, "r").read().splitlines()

from os import listdir
from os.path import isfile, join

files = listdir(anno_dir)
files = transformer.filter_func(files, lambda name : 'rating' in name)

def get_id(file_name):
    idx = file_name.index("rating")
    return file_name[:idx - 1]

# check completeness
anno_ids = transformer.map_func(files, lambda file_name : get_id(file_name))
anno_ids.sort()

train_test_ids = train_id_dataset + test_id_dataset
train_test_ids.sort()
assert(anno_ids == train_test_ids)

id_to_file = transformer.map_func(files, lambda file : (get_id(file), file))
id_to_file = dict(id_to_file)

def extractor(anno_dir, id_to_file, paper_id):
    file_path = anno_dir + id_to_file[paper_id]
    label_sent_dataset = csv_handler.csv_readlines(file_path, delimit = '\t')
    
    indexed_result = transformer.indexleft_func(label_sent_dataset)
    final = transformer.map_func(indexed_result, lambda p : (paper_id + "_" + str(p[0]), p[1][1], p[1][0]))
    return final

tmp = extractor(anno_dir, id_to_file, 'r18RxrXlG')
transformer.print_rows(tmp, 3)
    
train_dataset = transformer.flatmap_func(train_id_dataset, lambda paper_id : extractor(anno_dir, id_to_file, paper_id))
csv_handler.csv_writelines(train_output_path, train_dataset)

test_dataset = transformer.flatmap_func(test_id_dataset, lambda paper_id : extractor(anno_dir, id_to_file, paper_id))
csv_handler.csv_writelines(test_output_path, test_dataset)
