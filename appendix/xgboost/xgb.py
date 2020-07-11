from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
file_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../../pyfunctor')
import csv_handler as csv_handler
import transform as transformer 
from sklearn import metrics
import time
import xgboost as xgb
from util.weight import WeightClassCSV 

start_time = time.time()

train_set = csv_handler.csv_readlines(sys.argv[1])
dev_set = csv_handler.csv_readlines(sys.argv[2])
log_file_path = sys.argv[3]
seed = int(sys.argv[4])
label_weight = WeightClassCSV(sys.argv[1]).get_weights(['0', '1'])

def sep(dataset):
    sents = transformer.map_func(dataset, lambda triplet: triplet[1])
    labels = transformer.map_func(dataset, lambda triplet: (int)(triplet[2]))
    return (sents, labels)

(X_train, y_train) = sep(train_set)
(X_dev, y_dev) = sep(dev_set)

xgb_clf = Pipeline([('vect', 
                      #CountVectorizer(ngram_range=(1, 2))),
                      CountVectorizer(ngram_range=(1, 2), analyzer='word', tokenizer = lambda doc : doc.split(), token_pattern=r"*")),
                                          ('tfidf', TfidfTransformer()),
                                          #('xgboost', xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight = label_weight[1], num_parallel_tree = 5)),
                                          ('xgboost', xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight = label_weight[1], num_estimators = 5)),
                                          ])


xgb_clf.fit(X_train, y_train)
train_finish_time = time.time()
train_duration =  train_finish_time - start_time
print("train time is " + str(train_finish_time - start_time))

predicted = xgb_clf.predict(X_dev)

test_duration = time.time() - train_finish_time
print("test time is " + str(time.time() - train_finish_time))
print(metrics.classification_report(y_dev, predicted))

# output metric: precision,recall,f1,train_time, test_time
(precision, recall, fscore, support) = metrics.precision_recall_fscore_support(y_dev, predicted)
print("F1 is " + str(fscore))
#csv_handler.append_row(log_file_path, ['dataset', 'precision', 'recall', 'fscore', 'train_time', 'test_time'])
row = []
row.append(sys.argv[1])
row.append(precision[1])
row.append(recall[1])
row.append(fscore[1])
row.append(train_duration)
row.append(test_duration)
csv_handler.append_row(log_file_path, row)


