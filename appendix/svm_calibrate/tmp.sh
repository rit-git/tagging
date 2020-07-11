for dataset in "book_spoiler/0.2dev/1m" 
#for dataset in "deeptip/sent" 
do
    SEED=12
    echo ${dataset}
    DATA_DIR="../../../tip/data/${dataset}"
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    OUTPUT_FILE="./output/book_spoiler_0.2dev_1m_pred.csv"

    python ./svm_predict.py ${TRAIN_FILE} ${DEV_FILE} ${OUTPUT_FILE} ${SEED}
done
