for dataset in "book_spoiler" 
do
    SEED=12
    #SEED=1000
    echo ${dataset}
    DATA_DIR="../../../tip/data/${dataset}"
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    OUTPUT_FILE="./output/book_spoiler.csv"

    python ./lr_predict.py ${TRAIN_FILE} ${DEV_FILE} ${OUTPUT_FILE} ${SEED}
done
