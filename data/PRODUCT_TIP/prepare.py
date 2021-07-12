import sys
import pdb

sys.path.insert(0, '../../pyfunctor')

import csv_handler as csv_handler
import transform as transform

dataset = csv_handler.csv_readlines('generate_tips_data.tsv', delimit='\t')

dataset = dataset[1:]

dataset = transform.map_func(range(len(dataset)), lambda i : [i, dataset[i][5], dataset[i][6]])

csv_handler.csv_writelines('./all.csv', dataset)

