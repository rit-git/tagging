CUDA=0

#CUDA_VISIBLE_DEVICES=${CUDA} python ../../model/bert_predict.py data/glassdoor/google/pos.csv ./model/SENT/best ./output/output.csv
#CUDA_VISIBLE_DEVICES=${CUDA} python -m pdb ../../model/bert_predict.py data/SUGG/test.csv ./model/SUGG/best ./output/output.csv

model_name=book_spoiler/undersample
input=../../data/book_spoiler/undersample/dev.csv
output=./output/book_spoiler_undersample_pred.csv
CUDA_VISIBLE_DEVICES=${CUDA} python ../../model/bert_predict.py ${input} ./model/${model_name}/best ${output}
