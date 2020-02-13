import sys
sys.path.insert(0, "../../pyfunctor")

dataset='subtask1-heterographic-test'
input_path = "./semeval2017_task7/data/test/" + dataset + ".xml"
gold_path = "./semeval2017_task7/data/test/" + dataset + ".gold"

import xml.etree.ElementTree as ET
tree = ET.parse(input_path)
root = tree.getroot()

root.tag
sents = []
for child in root:
    tid = child.attrib['id']
    sentence = ""

    num_words = len(child)
    for i in range(num_words):
        sentence += child[i].text
        if i < num_words - 2:
            sentence += " "
    sents.append((tid, sentence))

import csv_handler as csv_handler
golds = csv_handler.csv_readlines(gold_path, delimit='\t')

import transform as transformer
assert(len(sents) == len(golds))
for i in range(len(sents)):
    assert(sents[i][0] == golds[i][0])
final = transformer.map_func(range(len(sents)), lambda i : (sents[i][0], sents[i][1], golds[i][1]))

import csv_handler as csv_handler
class CSV_Split(csv_handler.CSV_Handler):
    def __init__(self, dataset, seed = 0):
        self.seed = seed
        self.dataset = dataset

splitter = CSV_Split(final)
splitter.csv_split(0.2, "dev.csv", "train.csv")
