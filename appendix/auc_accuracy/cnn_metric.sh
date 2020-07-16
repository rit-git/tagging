mkdir result
CUDA=3
#for dataset in "SUGG" "HOTEL" "SENT" "PARA" "FUNNY_400K" "HOMO" "HETER" "TV" "BOOK_400K" "EVAL" "REQ" "FACT" "REF" "QUOTE" "ARGUE" "SUPPORT" "AGAINST" "AMAZON_400K" "YELP" "FUNNY_STAR" "BOOK_STAR" 
for dataset in "SUGG"
do
    DATA_DIR="../../data/${dataset}"
    echo ${dataset}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/cnn.csv"

    CUDA_VISIBLE_DEVICES=${CUDA} python ./train_nn_metric.py --dataset ${DATA_DIR} --model CNN --num_epoch 10 --log_file ${LOG_FILE}
done
