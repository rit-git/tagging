mkdir result
echo "Dataset,precision,recall,f1,train_time(s),test_time(s)" > result/lr.csv
#for random_seed in 1000 2000 3000 
#for dataset in "yelp_funny_balanced"
for dataset in "amazon_polarity" "yelp_polarity" "yelp_funny_balanced" "book_spoiler_balanced"
do
    #DATA_DIR="../../../tip/data/yelp_funny/0.2dev/2k"
    DATA_DIR="../../../tip/data/${dataset}"
    SEED=12
    echo ${DATA_DIR}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/xgb2.csv"

    python ./xgb.py ${TRAIN_FILE} ${DEV_FILE} ${LOG_FILE} ${SEED}
done
