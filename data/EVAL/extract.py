import sys
sys.path.insert(0, "../../pyfunctor")

import csv_handler as csv_handler
import transform as transformer

#usage: python type_extractor.py [type] 

typename = sys.argv[1]
# typename = 'evaluation' # options: evaluation, request, fact, reference, quote, non-arg
assert(typename in {'evaluation', 'request', 'fact', 'reference', 'quote'})

def extract(data, typename):
    dataset = csv_handler.csv_readlines("../EVAL/origin/" + data + "set.csv")
    dataset = transformer.indexleft_func(dataset)
    dataset = transformer.map_func(dataset, lambda row : (row[0], row[1][1], row[1][2]))
    output_path = "./" + data + ".csv"

    def e_func(triplet):
        label = 0
        if triplet[2] == typename:
            label = 1
        return (triplet[0], triplet[1], label)


    final = transformer.map_func(dataset, lambda triplet: e_func(triplet))
    csv_handler.csv_writelines(output_path, final)

extract('train', typename)
extract('dev', typename)

