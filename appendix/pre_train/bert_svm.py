from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
file_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, '../../pyfunctor')
import csv_handler as csv_handler
import transform as transformer 
import threads.transform as mtransformer 

from sklearn import metrics
from bert_embedding import BertEmbedding 
import numpy as np
import time

start_time = time.time()

train_set = csv_handler.csv_readlines(sys.argv[1])
dev_set = csv_handler.csv_readlines(sys.argv[2])
max_seq_length = int(sys.argv[3])
log_file_path = sys.argv[4]

class Embedder:
    def __init__(self, max_seq_length, batch_size = 32):
        self.embedder = BertEmbedding(max_seq_length)
        self.batch_size = batch_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        #result = []
        #for i, row in enumerate(X):
        #    embedding = self.embedder.project(row)
        #    result.append(embedding)

        # batching
        result = []
        i = 0

        while i < len(X):
            print("start processing {} / {}".format(i, len(X)))
            batch = X[i:(i + self.batch_size)] 
            embedding = self.embedder.project_batch(batch)

            result += embedding

            i += self.batch_size

        return np.array(result)

def sep(dataset):
    sents = transformer.map_func(dataset, lambda triplet: triplet[1])
    labels = transformer.map_func(dataset, lambda triplet: (int)(triplet[2]))
    return (sents, labels)

(X_train, y_train) = sep(train_set)
(X_dev, y_dev) = sep(dev_set)

text_clf = Pipeline([
    ('vect', Embedder(max_seq_length)),
    ('clf', LinearSVC(class_weight='balanced'))])

text_clf.fit(X_train, y_train)
train_finish_time = time.time()
train_duration =  train_finish_time - start_time
print("train time is " + str(train_finish_time - start_time))


predicted = text_clf.predict(X_dev)
test_duration = time.time() - train_finish_time
print("test time is " + str(time.time() - train_finish_time))

print(metrics.classification_report(y_dev, predicted))

# output metric: precision,recall,f1,train_time, test_time
(precision, recall, fscore, support) = metrics.precision_recall_fscore_support(y_dev, predicted)
#csv_handler.append_row(log_file_path, ['dataset', 'precision', 'recall', 'fscore', 'train_time', 'test_time'])
row = []
row.append(sys.argv[1])
row.append(precision[1])
row.append(recall[1])
row.append(fscore[1])
row.append(train_duration)
row.append(test_duration)
csv_handler.append_row(log_file_path, row)
