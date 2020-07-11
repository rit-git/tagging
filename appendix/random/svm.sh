mkdir result
echo "Dataset,precision,recall,f1,train_time(s),test_time(s)" > result/lr.csv
#for dataset in "yelp_funny/random/20k" 
for random_seed in 1000 2000 3000
do
    #DATA_DIR="../../../tip/data/book_spoiler/undersample"
    DATA_DIR="../../../tip/data/yelp_funny/random/1m_${random_seed}"
    SEED=12
    echo ${DATA_DIR}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/svm.csv"

    python ./svm.py ${TRAIN_FILE} ${DEV_FILE} ${LOG_FILE} ${SEED}
done
