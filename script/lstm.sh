CUDA=3
mkdir result
echo "Dataset,precision,recall,f1,train_time(s),test_time(s)" > result/lstm.csv
for dataset in "SUGG" 
do
    echo ${dataset}
    DATA_DIR="../../tip/data/${dataset}"
    TRAIN_FILE=${DATA_DIR}/train.csv
    DEV_FILE=${DATA_DIR}/dev.csv
    LOG_FILE="./result/lstm.csv"

    # AMAZON uses up memory. Append parameters: --glove 6B --emb_size 100 
    CUDA_VISIBLE_DEVICES=${CUDA} python ../model/train_nn.py --dataset ${DATA_DIR} --model LSTM --num_epoch 10 --log_file ${LOG_FILE}
done
