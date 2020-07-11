CUDA=2

#!/bin/bash
#for dataset in "semeval2019task9" "hotel" "deeptip/sent" "deeptip/para" 
#for dataset in "semeval2017task7/homographic" "semeval2017task7/heterographic" "tv_spoiler" 
#for dataset in "naacl19_argument/evaluation" "naacl19_argument/request" "naacl19_argument/fact" "naacl19_argument/reference" "naacl19_argument/quote"
#for dataset in "emnlp18_argument" "emnlp18_argument/support" "emnlp18_argument/opposite"
#for dataset in "yelp_funny" "book_spoiler"
#for dataset in "yelp_funny_balanced" "book_spoiler_balanced"
#for dataset in "semeval2019task9"
#for dataset in  "naacl19_argument/request" "naacl19_argument/reference" 
for dataset in "book_spoiler_balanced/data_size/400k"
do
    echo ${dataset}
    DATA_DIR="../../../tip/data/${dataset}"
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    BERT_SEQ_LENGTH=64
    LOG_FILE='./result/tmp.txt'

    CUDA_VISIBLE_DEVICES=${CUDA} python bert_svm.py ${TRAIN_FILE} ${DEV_FILE} ${BERT_SEQ_LENGTH} ${LOG_FILE}
done
