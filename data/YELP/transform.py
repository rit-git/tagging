import sys
sys.path.insert(0, '../../pyfunctor')
import transform as transformer
import csv_handler as csv_handler

input_file=sys.argv[1]
output_file=sys.argv[2]
dataset = csv_handler.csv_readlines(input_file)

dataset = transformer.indexleft_func(dataset)
final = transformer.map_func(dataset, lambda p: (p[0], p[1][2], int(p[1][0]) - 1))

csv_handler.csv_writelines(output_file, final)
