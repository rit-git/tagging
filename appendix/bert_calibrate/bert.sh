CUDA=3

for dataset in "book_spoiler"
do
    DATA_DIR="/home/ubuntu/users/jinfeng/tip/data/${dataset}"
    MODEL_DIR="./model/${dataset}"
    LOG_FILE_PATH="./log/bert.log"
    echo "" >> ${LOG_FILE_PATH}
    SEED=1000

    CUDA_VISIBLE_DEVICES=${CUDA} python ../../model/transformer.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name SST-2 \
        --do_train \
        --do_lower_case \
        --data_dir $DATA_DIR \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=32   \
        --per_gpu_train_batch_size=32   \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODEL_DIR \
        --log_file_path $LOG_FILE_PATH \
        --save_snapshots 1 \
        --seed ${SEED}
done
