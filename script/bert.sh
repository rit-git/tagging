CUDA=1
mkdir result
mkdir model
echo "Dataset,precision,recall,f1,train_time(s),test_time(s)" > result/bert.csv
for dataset in "SUGG" 
do
    dataset="${dataset}"
    export TASK_NAME=SST-2
    DATA_DIR="../data/${dataset}"
    LOG_DIR="./model/${dataset}"
    LOG_FILE_PATH="./result/bert.csv"

    rm -rf $LOG_DIR

    CUDA_VISIBLE_DEVICES=${CUDA} python ../model/bert.py \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $DATA_DIR \
        --bert_model bert-base-uncased \
        --max_seq_length 128 \
        --train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $LOG_DIR \
        --log_file_path $LOG_FILE_PATH

done
