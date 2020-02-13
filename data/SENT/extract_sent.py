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
sent_n = csv_handler.csv_readlines(input_dir + "sent_tip.neg", delimit = '\t', quoter = csv.QUOTE_NONE)
sent_n = transformer.map_func(sent_n, lambda p : (p[0], p[1], '0'))

sent_p = csv_handler.csv_readlines(input_dir + "sent_tip.pos", delimit = '\t', quoter = csv.QUOTE_NONE)
sent_p = transformer.map_func(sent_p, lambda p : (p[0], p[1], '1'))

sent = sent_n + sent_p
for row in sent:
    assert(len(row) == 3)
ids = set(transformer.map_func(sent, lambda p : p[0]))
assert(len(ids) == len(sent))

splitter = CSV_Split(sent)
splitter.csv_split(0.2, "dev.csv", "train.csv")
