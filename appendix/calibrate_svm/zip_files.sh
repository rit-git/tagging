dataset="SUGG"
first_csv_file="./output/${dataset}_pred.csv"
first_idxs='1'
second_csv_file="../../data/${dataset}/dev.csv"
second_idxs='2'
output_csv_file="./output/${dataset}_zip.csv"
python ../calibrate/zip_files.py ${first_csv_file} ${first_idxs} ${second_csv_file} ${second_idxs} ${output_csv_file}

