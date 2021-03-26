import pdb

import sys
import click
import os
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

if 'TAGGING_HOME' in os.environ:
    pyfunctor_path = os.environ['TAGGING_HOME'] + "/pyfunctor"
    sys.path.append(pyfunctor_path)
else:
    sys.exit("please declara environment variable 'TAGGING_HOME'")

import csv_handler as csv_handler
import transform as transform
from nlp.bert_predict import BertModel
from nlp.bert_predict import bert_estimate

@click.group()
def cli():
    pass

@click.command()
@click.argument('input_path')
@click.argument('col_true')
@click.argument('col_pred')
@click.option('-m', '--metric', default='f1', help='Specify an evaluation metric in {accuracy, cohen, f1, quad}. quad means precision_recall_fscore_support')
@click.option('-o', '--output_path', default = "", help='Write output to a file instead of stdout')
@click.option('-w', '--with_header', is_flag=True, help='If set, the first row will be ignored')
def evaluate(input_path, col_true, col_pred, metric, output_path, with_header):
    '''evaluate the quality of predictions with a metric (f1 by default), and output the metric scores'''

    result = []
    dataset = csv_handler.csv_readlines(input_path)
    if with_header == True:
        dataset = dataset[1:]

    col_true = int(col_true) - 1
    col_pred = int(col_pred) - 1
    y_true = transform.map_func(dataset, lambda row : int(row[col_true]))
    y_pred = transform.map_func(dataset, lambda row : int(row[col_pred]))

    def check_validity(class_array):
        for cls in class_array:
            assert(cls == 0 or cls == 1)
    check_validity(y_true)
    check_validity(y_pred)

    support_set = {'f1', 'accuracy', 'cohen', 'quad'}
    if metric not in support_set:
        sys.exit('please specify a valid metric in terms of f1, accuracy, cohen, or quad (i.e. precision_recall_fscore_support)')
    elif metric == 'f1':
        result.append(['f1'])
        result.append([f1_score(y_true, y_pred)])
    elif metric == 'accuracy':
        result.append(['accuracy'])
        result.append([accuracy_score(y_true, y_pred)])
    elif metric == 'cohen':
        result.append([cohen_kappa_score(y_true, y_pred)]) 
    elif metric == 'quad':
        (precision, recall, fscore, support) = precision_recall_fscore_support(y_true, y_pred)
        result.append(['class', 'precision', 'recall', 'fscore', 'support'])
        result.append([0, precision[0], recall[0], fscore[0], support[0]])
        result.append([1, precision[1], recall[1], fscore[1], support[1]])

    csv_handler.csv_writelines(output_path, result)

@click.command()
@click.argument('input_path')
@click.argument('model_dir')
@click.option('-c', '--text_col', default = 2, help='Specify the column of texts, 2 by default')
@click.option('-o', '--output_path', default = "", help='Write output to a file instead of stdout')
@click.option('-g', '--gpu',  default = "0", help='Assign a GPU for estimation')
@click.option('-w', '--with_header', is_flag=True, help='If set, the first row will be ignored')
def estimate(input_path, model_dir, text_col, output_path, gpu, with_header):
    '''output negative-class probability, positive-class probability and predicted argmax class '''

    bert_estimate(input_path, text_col, output_path, model_dir, gpu, with_header)

@click.command()
@click.argument('input_path')
@click.argument('output_model_dir')
@click.option('-tc', '--text_col', default = 2, help='Specify the column of texts, 2 by default')
@click.option('-lc', '--label_col', default = 3, help='Specifcy the column of labels, 3 by default')
@click.option('-m', '--model_dir', default='bert-base-uncased', help='Specifcy a source model to start with or otherwise bert-base-uncased')
@click.option('-g', '--gpu',  default = "0", help='Assign a GPU for estimation')
@click.option('-w', '--with_header', is_flag=True, help='If set, the first row will be ignored')
def finetune(input_path, output_model_dir, text_col, label_col, model_dir, gpu, with_header):
    '''Train a new model or finetune an existing model with labels, output fine-tuned model'''

    # assign GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    dataset = csv_handler.csv_readlines(input_path)
    header = None
    if with_header == True:
        header = dataset[0]
        dataset = dataset[1:]

    print("Loading source model from %s ...\n" % (model_dir))
    model = BertModel(model_dir)

    text_col = text_col - 1
    label_col = label_col - 1

    labels = transform.map_func(range(len(dataset)), lambda i : [i, dataset[i][text_col], dataset[i][label_col]])

    print("Fine-tuning with input labels")
    model.train(labels)

    model.checkpoint(output_model_dir)

    print("Finished. Fine-tuned model is ready at " + output_model_dir)

cli.add_command(evaluate)
cli.add_command(estimate)
cli.add_command(finetune)
