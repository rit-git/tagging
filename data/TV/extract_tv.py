import sys
sys.path.insert(0, "../../pyfunctor")
import csv_handler as csv_handler
import transform as transformer

def transform(input_path, output_path):
    dataset = csv_handler.csv_readlines(input_path)
    dataset = dataset[1:]

    dataset = transformer.map_func(range(len(dataset)), lambda i: (i, dataset[i][0], 1 if dataset[i][1] == "True" else 0))
    csv_handler.csv_writelines(output_path, dataset)

  
transform("./spoilers/train.balanced.csv", "train.csv")  
transform("./spoilers/test.balanced.csv", "dev.csv")  
