for dataset in "yelp_funny/undersample" 
#for dataset in "deeptip/sent" 
do
    SEED=12
    echo ${dataset}
    DATA_DIR="../../../tip/data/${dataset}"
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    OUTPUT_FILE="./output/yelp_funny_undersample_pred.csv"

    python -m pdb ../svm_calibrate/svm_predict.py ${TRAIN_FILE} ${DEV_FILE} ${OUTPUT_FILE} ${SEED}
done
