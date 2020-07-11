for dataset in "book_spoiler/undersample" 
do
    SEED=12
    #SEED=1000
    echo ${dataset}
    DATA_DIR="../../../tip/data/${dataset}"
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    OUTPUT_FILE="./output/book_spoiler_undersample_pred.csv"

    python ../lr_calibrate/lr_predict.py ${TRAIN_FILE} ${DEV_FILE} ${OUTPUT_FILE} ${SEED}
done
