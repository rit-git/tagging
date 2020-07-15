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
from sklearn import metrics
import time

start_time = time.time()

train_set = csv_handler.csv_readlines(sys.argv[1])
dev_set = csv_handler.csv_readlines(sys.argv[2])

output_path = sys.argv[3]
if os.path.exists(output_path):
    os.remove(output_path)

seed = int(sys.argv[4])

def sep(dataset):
    sents = transformer.map_func(dataset, lambda triplet: triplet[1])
    labels = transformer.map_func(dataset, lambda triplet: (int)(triplet[2]))
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
csv_handler.append_row(output_path, ['score_1', 'predict', 'text', 'ground'])
result = []
for i in range(len(predicted_proba)):
    score_1 = predicted_proba[i]
    predict = predicted[i]
    text = X_dev[i]
    ground = y_dev[i]
    result.append([score_1, predict, text, ground])
csv_handler.csv_writelines(output_path, result)
