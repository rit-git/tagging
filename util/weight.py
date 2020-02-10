import os
file_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, file_path + '/../pyfunctor')
import csv_handler as csv_handler
import transform as transform
def weight_class(labels):
    if len(labels) == 0:
        return {}
    result = {}
    for lb in set(labels):
        result[lb] = 0
    for lb in labels:
        result[lb] += 1
    for key in result:
        # plan 1
        #result[key] = result[key] / len(labels)
        #result[key] = 1- result[key]

        # plan 2 sklearn
        result[key] = len(labels) / 2.0 / result[key]
    return result

class WeightClassCSV:
    def __init__(self, csv_file_path):
        dataset = csv_handler.csv_readlines(csv_file_path)
        labels = transform.map_func(dataset, lambda t : t[2])
        self.weight_map = weight_class(labels)

    def get_weights(self):
        return self.weight_map

    def get_weights(self, targets):
        return transform.map_func(targets, lambda lb : self.weight_map[lb])
