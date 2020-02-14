import sys
sys.path.insert(0, "../../pyfunctor")

import csv_handler as csv_handler
import transform as transformer

#usage: python type_extractor.py [type] 

typename = sys.argv[1]
# typename = 'evaluation' # options: evaluation, request, fact, referenc, quote, non-arg
assert(typename in {'Argument_against', 'Argument_for', 'NoArgument'})

def extract(data, typename):
    dataset = csv_handler.csv_readlines("./dataset/" + data + "_raw.csv")
    output_path = "./" + data + ".csv"

    def func_1(triplet):
        label = 0
        if triplet[2] == typename:
            label = 1
        return (triplet[0], triplet[1], label)

    e_func = func_1

    def func_2(triplet):
        label = 1
        if triplet[2] == typename:
            label = 0
        return (triplet[0], triplet[1], label)

    if typename == 'NoArgument':
        e_func = func_2

    final = transformer.map_func(dataset, lambda triplet: e_func(triplet))
    csv_handler.csv_writelines(output_path, final)

extract('train', typename)
extract('dev', typename)

