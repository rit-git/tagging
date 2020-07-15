import sys
from os import listdir
sys.path.insert(0, "../../pyfunctor")
import csv_handler as csv_handler
import transform as transform


file_set = sys.argv[1]
input_dir = "./" + file_set 
output_path = file_set + ".csv"

def distill(input_path):
    dataset = csv_handler.csv_readlines(input_path, delimit='\t')
    return dataset

final = []
for file_name in listdir(input_dir):
    dataset = distill(input_dir + "/" + file_name)
    final += dataset 

final = transform.indexleft_flat_func(final)
csv_handler.csv_writelines(output_path, final)
