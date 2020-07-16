CUDA=3

#CUDA_VISIBLE_DEVICES=${CUDA} python ../../model/bert_predict.py data/glassdoor/google/pos.csv ./model/SENT/best ./output/output.csv
#CUDA_VISIBLE_DEVICES=${CUDA} python -m pdb ../../model/bert_predict.py data/SUGG/test.csv ./model/SUGG/best ./output/output.csv

mkdir output
dataset=SUGG
input=../../data/${dataset}/dev.csv
output=./output/${dataset}_pred.csv
CUDA_VISIBLE_DEVICES=${CUDA} python ../../model/bert_predict.py ${input} ./model/${dataset}/best ${output}
