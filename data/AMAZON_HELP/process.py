import os
import sys
import json
sys.path.insert(0, os.environ['PYFUNCTOR_HOME'])
import transform as transform
import csv_handler as csv_handler
import json_handler as json_handler

from statistics import median

if __name__=="__main__":
    json_input_path=sys.argv[1]
    csv_output_path=sys.argv[2]

    dataset = json_handler.load_json_dicts(json_input_path)
    triplet = transform.map_func(dataset, lambda d : [d['asin'], d['sentence'], float(d['helpful'])])

    helpful = transform.map_func(triplet, lambda t : t[2])
    med = median(helpful)

    final = transform.map_func(triplet, lambda t : [t[0], t[1], 1 if t[2] > med else 0])

    csv_handler.csv_writelines(csv_output_path, final)
