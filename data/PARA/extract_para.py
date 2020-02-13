import sys
sys.path.insert(0, "../../pyfunctor")
import csv_handler as csv_handler
import transform as transformer
import csv

class CSV_Split(csv_handler.CSV_Handler):
    def __init__(self, dataset, seed = 0):
        self.seed = seed
        self.dataset = dataset

input_dir = "./DEEPTip/dataset/"
# paragraph
para_n = csv_handler.csv_readlines(input_dir + "para_tip.neg", delimit = '\t', quoter = csv.QUOTE_NONE)
para_n = transformer.map_func(para_n, lambda p : (p[0], p[1], '0'))

para_p = csv_handler.csv_readlines(input_dir + "para_tip.pos", delimit = '\t', quoter = csv.QUOTE_NONE)
para_p = transformer.map_func(para_p, lambda p : (p[0], p[1], '1'))

para = para_n + para_p
for row in para:
    assert(len(row) == 3)
ids = set(transformer.map_func(para, lambda p : p[0]))
assert(len(ids) == len(para))

splitter = CSV_Split(para)
splitter.csv_split(0.2, "dev.csv", "train.csv")
