from sampler import Sampler

import transform as transformer
import csv
import sys
import os
csv.field_size_limit(sys.maxsize)

def csv_readlines(data_path, encoder = 'utf-8', delimit = ',', quoter=csv.QUOTE_MINIMAL):
    reader = csv.reader(open(data_path, 'rt', encoding = encoder), delimiter = delimit, quoting = quoter)
    dataset = []
    for row in reader:
        dataset.append(row)
    return dataset
   
def csv_writelines(output_path, dataset, encoder = 'utf-8', delimit = ','):
    if output_path != "":
        writer = csv.writer(open(output_path, 'w', encoding = encoder), delimiter = delimit)
    else:
        writer = csv.writer(sys.stdout)
    for row in dataset:
        writer.writerow(row)

def append_row(csv_file_name, row, encoder = 'utf-8', delimit = ','):
    if not os.path.exists(csv_file_name):
        f = open(csv_file_name, "w+", encoding = encoder)
        f.close()

    csv_writer = csv.writer(open(csv_file_name, 'a', encoding = encoder), delimiter = delimit)
    csv_writer.writerow(row)

class CSV_Handler:
    def __init__(self, data_path, seed = 0, encoder = 'utf-8', delimit = ','):
        self.seed = seed
        self.dataset = csv_readlines(data_path, encoder, delimit)  

    def csv_shuf(self, num_samples, output_path):
        assert(len(self.dataset) >= num_samples)
        sampled_idx = self.__sampled_idx(num_samples)
        samples = transformer.map_func(sampled_idx, lambda idx : self.dataset[idx])
        csv_writelines(output_path, samples)

    def csv_split(self, percentage, first_output_path="", second_output_path=""):
        assert(percentage > 0)
        assert(percentage < 1)
        first_size = percentage * len(self.dataset)
        first_idx = self.__sampled_idx(first_size)

        # first output
        first_dataset = transformer.map_func(first_idx, lambda idx : self.dataset[idx])

        if first_output_path != "":
            csv_writelines(first_output_path, first_dataset)

        # second output
        first_idx_set = set(first_idx)
        second_idx = transformer.filter_func(range(len(self.dataset)), lambda idx : idx not in first_idx_set)
        second_dataset = transformer.map_func(second_idx, lambda idx : self.dataset[idx])

        if second_output_path != "":
            csv_writelines(second_output_path, second_dataset)

        return (first_dataset, second_dataset)

    def __sampled_idx(self, num_samples):
        sampler = Sampler(self.seed)
        idx_set = sampler.distinct_ints(num_samples, 0, len(self.dataset) - 1)
        idx_set = list(idx_set)
        idx_set.sort(key = lambda n : n)
        return idx_set
        
        
