import sys
import json

root_directory = '../../'
sys.path.insert(0, root_directory + "pyfunctor")

import transform as transformer
import csv_handler as csv_handler

input_path = "./goodreads_reviews_spoiler.json"
output_path = "spoiler.csv"

fin = open(input_path, "r")
dataset = []
for row in fin.readlines():
    dataset.append(row)

json_dataset = transformer.map_func(dataset, lambda line: json.loads(line))

def format_func(line):
    num_true = line.count('"has_spoiler": true,')
    num_false = line.count('"has_spoiler": false,')
    if num_true == 0:
        assert(num_false == 1)
        line = line.replace('"has_spoiler": false,', '"has_spoiler": "false",')
    elif num_true == 1:
        assert(num_false == 0)
        line = line.replace('"has_spoiler": true,', '"has_spoiler": "true",')
    else:
        assert(False)
        
    line = json.loads(line)
    return line
        
json_dataset = transformer.map_func(dataset, lambda line : format_func(line))

id_sents = transformer.map_func(json_dataset, lambda jsn : (jsn['review_id'], jsn['review_sentences']))

def flat_sents_func(review_id, sents):
    result = []
    for i in range(len(sents)):
        sent_id = review_id + "###" + str(i)
        sent_text = sents[i][1]
        sent_label = sents[i][0]
        result.append((sent_id, sent_text, sent_label))
    return result

final_sents = transformer.flatmap_func(id_sents, lambda p : flat_sents_func(p[0], p[1]))

csv_handler.csv_writelines(output_path, final_sents)
