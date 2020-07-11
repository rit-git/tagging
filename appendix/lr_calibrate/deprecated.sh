#currently only apply to bert
input_csv="./output/book_spoiler_400k_pred.csv"
y_true_col=4 # ground_truth 0 or 1 (desired)
y_pred_col=1 # bert pos predicted score, before softmax
num_threds=200 # number of thresholds to try
output_csv="./tmp.csv"
python -m pdb ../../../tip/model/bert_calibrate.py ${input_csv} ${y_true_col} ${y_pred_col} ${num_threds} ${output_csv}
