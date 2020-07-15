##currently only apply to bert
#input_csv="./output.csv"
#y_true_col=1 # ground_truth 0 or 1 (desired)
#y_pred_col=0 # bert pos predicted score, before softmax
#num_threds=200 # number of thresholds to try
#output_csv="./tmp.csv"
#python ../../model/bert_calibrate.py ${input_csv} ${y_true_col} ${y_pred_col} ${num_threds} ${output_csv}


#currently only apply to bert
for num_threds in 100 200 300 400 
do
    input_csv="./output.csv"
    y_true_col=1 # ground_truth 0 or 1 (desired)
    y_pred_col=0 # bert pos predicted score, before softmax
    output_csv="./tmp.csv"
    print_header=0
    python ../../model/bert_calibrate.py ${input_csv} ${y_true_col} ${y_pred_col} ${num_threds} ${output_csv} ${print_header}
    q -d , 'select max(c4) from tmp.csv'
done
