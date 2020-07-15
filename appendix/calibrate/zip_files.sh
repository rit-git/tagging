#LR
#first_csv_file=../../../tagging/appendix/calibrate/output/book_spoiler_0.2dev_1m_pred.csv
first_csv_file=../../../tagging/appendix/lr_subsample/output/book_spoiler_undersample_pred.csv
first_idxs='1'
second_csv_file=../../data/book_spoiler/undersample/dev.csv
second_idxs='2'
output_csv_file=./output.csv
python -m pdb zip_files.py ${first_csv_file} ${first_idxs} ${second_csv_file} ${second_idxs} ${output_csv_file}

##SVM
#first_csv_file=../../../tagging/appendix/svm_calibrate/output/book_spoiler_undersample_pred.csv
#first_idxs='0'
#second_csv_file=../../data/book_spoiler/undersample/dev.csv
#second_idxs='2'
#output_csv_file=./output.csv
#python -m pdb zip_files.py ${first_csv_file} ${first_idxs} ${second_csv_file} ${second_idxs} ${output_csv_file}
