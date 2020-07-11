mkdir result
echo "Dataset,precision,recall,f1,train_time(s),test_time(s)" > result/lr.csv
for dataset in "book_spoiler/undersample" 
#for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#for i in 1
do
    #DATA_DIR="../../../tip/data/book_spoiler/undersample"
    DATA_DIR="../../../tip/data/${dataset}"
    SEED=12
    echo ${dataset}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/svm.csv"

    python ./svm_subsample.py ${TRAIN_FILE} ${DEV_FILE} ${LOG_FILE} ${SEED}
done
