for dataset in "book_spoiler/undersample" 
#for dataset in "deeptip/sent" 
do
    OUTPUT_FILE="./output/book_spoiler_undersample_pred.csv"
    DATA_DIR="../../../tip/data/${dataset}"

    SEED=12
    echo ${dataset}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv

    python -m pdb ./svm_predict.py ${TRAIN_FILE} ${DEV_FILE} ${OUTPUT_FILE} ${SEED}
done
