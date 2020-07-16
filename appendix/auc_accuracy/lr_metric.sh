mkdir result
#echo "Dataset,precision,recall,f1, auc, accuracy" > result/lr.csv
for dataset in "SUGG" 
#for dataset in "SUGG" "HOTEL" "SENT" "PARA" "FUNNY_400K" "HOMO" "HETER" "TV" "BOOK_400K" "EVAL" "REQ" "FACT" "REF" "QUOTE" "ARGUE" "SUPPORT" "AGAINST" "AMAZON_400K" "YELP" "FUNNY_STAR" "BOOK_STAR" 
do
    DATA_DIR="../../../tagging/data/${dataset}"
    SEED=12
    echo ${dataset}
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/lr.csv"

    python ./lr_metric.py ${TRAIN_FILE} ${DEV_FILE} ${LOG_FILE} ${SEED}
done
