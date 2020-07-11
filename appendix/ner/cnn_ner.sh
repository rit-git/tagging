mkdir result
CUDA=2
#echo "Dataset,precision,recall,f1, auc, accuracy" > result/lr.csv
#for dataset in "semeval2019task9" "yelp_funny/data_size/400k" "book_spoiler/data_size/400k" "yelp_funny_balanced"
#for dataset in "SUGG" "HOTEL" "SENT" "PARA" "FUNNY_400K" "HOMO" "HETER" "TV" "BOOK_400K" "EVAL" "REQ" "FACT" "REF" "QUOTE" "ARGUE" "SUPPORT" "AGAINST" "AMAZON_400K" "YELP" "FUNNY_STAR" "BOOK_STAR" 
for dataset in "SEMEVAL20_TASK6_NER"
#for dataset in "tmp"
do
    DATA_DIR="../../../tagging/data/${dataset}"
    echo ${dataset}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/cnn.csv"

    CUDA_VISIBLE_DEVICES=${CUDA} python ./train_nn_ner.py --dataset ${DATA_DIR} --model CNN --num_epoch 10 --log_file ${LOG_FILE}
done
