mkdir result
for dataset in "SUGG"
do
    DATA_DIR="../../data/${dataset}"
    SEED=12
    echo ${DATA_DIR}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/naive_bayes.csv"

    python ./naive_bayes.py ${TRAIN_FILE} ${DEV_FILE} ${LOG_FILE} ${SEED}
done
