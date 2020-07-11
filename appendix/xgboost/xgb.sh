mkdir result
echo "Dataset,precision,recall,f1,train_time(s),test_time(s)" > result/lr.csv
#for random_seed in 1000 2000 3000 
#for dataset in "yelp_funny_balanced"
#for dataset in "semeval2019task9" "hotel" "deeptip/sent" "deeptip/para" "yelp_funny" "semeval2017task7/homographic" "semeval2017task7/heterographic" "tv_spoiler" "book_spoiler" "naacl19_argument/evaluation" "naacl19_argument/request" "naacl19_argument/fact" "naacl19_argument/reference" "naacl19_argument/quote" "emnlp18_argument" "emnlp18_argument/support" "emnlp18_argument/opposite" "amazon_polarity" "yelp_polarity" "yelp_funny_balanced" "book_spoiler_balanced"
for dataset in "semeval2019task9"
do
    #DATA_DIR="../../../tip/data/yelp_funny/0.2dev/2k"
    DATA_DIR="../../../tip/data/${dataset}"
    SEED=12
    echo ${DATA_DIR}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/xgb.csv"

    python ./xgb.py ${TRAIN_FILE} ${DEV_FILE} ${LOG_FILE} ${SEED}
done
