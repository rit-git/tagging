import sys
sys.path.insert(0, "../../pyfunctor")

import transform as transformer
import csv_handler as csv_handler

class CSV_Split(csv_handler.CSV_Handler):
    def __init__(self, dataset, seed = 0):
        self.seed = seed
        self.dataset = dataset

input_path = sys.argv[1]
train_output_path = "train.csv"
dev_output_path = "dev.csv"

dataset = csv_handler.csv_readlines(input_path)
dataset = dataset[1:]

dataset = transformer.indexleft_func(dataset)
dataset = transformer.map_func(dataset, lambda pair : (pair[0], pair[1][0], pair[1][1]))

splitter = CSV_Split(dataset)
splitter.csv_split(0.2, "dev.csv", "train.csv")

