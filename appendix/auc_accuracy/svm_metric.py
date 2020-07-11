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
import transform as transform 
from sklearn import metrics
import time

start_time = time.time()

train_set = csv_handler.csv_readlines(sys.argv[1])
dev_set = csv_handler.csv_readlines(sys.argv[2])
log_file_path = sys.argv[3]

seed = int(sys.argv[4])

def sep(dataset):
    sents = transform.map_func(dataset, lambda triplet: triplet[1])
    labels = transform.map_func(dataset, lambda triplet: (int)(triplet[2]))
    return (sents, labels)

(X_train, y_train) = sep(train_set)
(X_dev, y_dev) = sep(dev_set)

text_clf = Pipeline([('vect', 
                      #CountVectorizer(ngram_range=(1, 2))),
                      CountVectorizer(ngram_range=(1, 2), analyzer='word', tokenizer = lambda doc : doc.split(), token_pattern=r"*")),
                                          ('tfidf', TfidfTransformer()),
                                          ('clf', LinearSVC(random_state=10000)),
                                          #('clf', LinearSVC(kernel='linear', class_weight='balanced', random_state=10000)),
                                          #('clf', LinearSVC(random_state=10000)),
                                          ])

text_clf.fit(X_train, y_train)

train_finish_time = time.time()
train_duration =  train_finish_time - start_time
print("train time is " + str(train_duration))


print("predicting...")
predicted = text_clf.predict(X_dev)
predicted_proba = text_clf.decision_function(X_dev)


assert(len(predicted_proba) == len(X_dev))
assert(len(X_dev) == len(y_dev))

print("logging...")

(precision, recall, fscore, support) = metrics.precision_recall_fscore_support(y_dev, predicted)

row = []
row.append(sys.argv[1])
row.append(precision[1])
row.append(recall[1])
row.append(fscore[1])

auc = metrics.roc_auc_score(y_dev, predicted_proba)
row.append(auc)

accuracy = metrics.accuracy_score(y_dev, predicted)
row.append(accuracy)

csv_handler.append_row(log_file_path, row)
