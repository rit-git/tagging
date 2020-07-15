import sys
sys.path.insert(0, "../../pyfunctor")

import transform as transform
import csv_handler as csv_handler

if __name__ == "__main__":
    result = []

    first_file=sys.argv[1]
    first_idx=sys.argv[2].split(',')
    first_idx = transform.map_func(first_idx, lambda idx : int(idx))

    second_file=sys.argv[3]
    second_idx=sys.argv[4].split(',')
    second_idx = transform.map_func(second_idx, lambda idx: int(idx))


    output_csv_file=sys.argv[5]

    first_dataset = csv_handler.csv_readlines(first_file)
    result = transform.map_func(first_dataset, lambda row : [row[idx] for idx in first_idx])

    second_dataset = csv_handler.csv_readlines(second_file)
    second_result = transform.map_func(second_dataset, lambda row : [row[idx] for idx in second_idx])

    assert(len(first_dataset) == len(second_dataset))

    final = transform.map_func(zip(result, second_result), lambda p : p[0] + p[1])

    csv_handler.csv_writelines(output_csv_file, final)
