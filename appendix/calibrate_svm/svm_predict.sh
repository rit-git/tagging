mkdir output
for dataset in "SUGG" 
do
    DATA_DIR="../../data/${dataset}"
    OUTPUT_FILE="./output/${dataset}_pred.csv"

    SEED=12
    echo ${dataset}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv

    python ./svm_predict.py ${TRAIN_FILE} ${DEV_FILE} ${OUTPUT_FILE} ${SEED}
done
