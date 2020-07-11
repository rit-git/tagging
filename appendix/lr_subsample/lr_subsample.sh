mkdir result
echo "Dataset,precision,recall,f1,train_time(s),test_time(s)" > result/lr.csv
for dataset in "yelp_funny/random/1m" 
do
    #DATA_DIR="../../../tip/data/${dataset}"
    DATA_DIR="../../../tip/util/model_selection/tmp-1000"
    SEED=12
    echo ${dataset}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/lr.csv"

    python ./lr_subsample.py ${TRAIN_FILE} ${DEV_FILE} ${LOG_FILE} ${SEED}
done
