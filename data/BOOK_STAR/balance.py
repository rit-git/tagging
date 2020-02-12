import sys
sys.path.insert(0, '../../pyfunctor')
import csv_handler as csv_handler
import transform as transform
from sampler import Sampler

csv_file = sys.argv[1]

output_file = "spoiler_balanced.csv"
if len(sys.argv) > 2:
    output_file = sys.argv[2]

def max_balancer(input_csv_path, output_csv_path='./output.csv'):
    dataset = csv_handler.csv_readlines(input_csv_path)

    pos_dataset = transform.filter_func(dataset, lambda row : row[2] == '1')
    neg_dataset = transform.filter_func(dataset, lambda row: row[2] == '0')

    assert(len(pos_dataset) <= len(neg_dataset))
    sampler = Sampler()
    neg_dataset = sampler.sample_rows(neg_dataset, len(pos_dataset))

    pos_ids = transform.map_func(pos_dataset, lambda row : row[0])
    neg_ids = transform.map_func(neg_dataset, lambda row : row[0])

    select_id_set = set(pos_ids + neg_ids)
    final = transform.filter_func(dataset, lambda row : row[0] in select_id_set)

    csv_handler.csv_writelines(output_csv_path, final)

max_balancer(csv_file, output_file)


