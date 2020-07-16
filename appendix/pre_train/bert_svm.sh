CUDA=3

for dataset in "SUGG"
do
    echo ${dataset}
    DATA_DIR="../../data/${dataset}"
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    BERT_SEQ_LENGTH=64
    LOG_FILE='./result/bert_svm.txt'

    CUDA_VISIBLE_DEVICES=${CUDA} python bert_svm.py ${TRAIN_FILE} ${DEV_FILE} ${BERT_SEQ_LENGTH} ${LOG_FILE}
done
