import sys
sys.path.insert(0, "../../pyfunctor")

import csv
import csv_handler as csv_handler
import transform as transformer

data_root = "./dataset"
data_dir = data_root + "/data/complete/"
topics = ['abortion', 'cloning', 'death_penalty', 'gun_control', 'marijuana_legalization', 'minimum_wage', 'nuclear_energy', 'school_uniforms']

dataset = []
for tp in topics:
    file_path = data_dir + tp + ".tsv"
    records = csv_handler.csv_readlines(file_path, delimit = '\t', quoter=csv.QUOTE_NONE)
    records = records[1:]
    
    def row_functor(i, records):
        assert(i < len(records))
        row = records[i]
        
        rid = row[0] + "_" + str(i)
        sent = row[4]
        label = row[5]
        split = row[6]
        
        return (rid, sent, label, split)
        
    records = transformer.map_func(range(len(records)), lambda i : row_functor(i, records))
    
    print(len(records))
    dataset = dataset + records[1:]
print(len(dataset))
print(dataset[0])

def select(split, dataset):
    final = transformer.filter_func(dataset, lambda row : row[3] == split)
    final = transformer.map_func(final, lambda row : (row[0], row[1], row[2]))
    return final

train_set = select("train", dataset)
print(len(train_set))

dev_set = select("test", dataset)
print(len(dev_set))

val_set = select("val", dataset)
print(len(val_set))

csv_handler.csv_writelines(data_root + "/train_raw.csv", train_set)
csv_handler.csv_writelines(data_root + "/dev_raw.csv", dev_set)
csv_handler.csv_writelines(data_root + "/val_raw.csv", val_set)
