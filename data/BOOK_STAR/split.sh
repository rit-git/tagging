DATA_DIR='./'
DATA_FILENAME='spoiler_balanced.csv'
SPLIT_PERCENT=0.2

#split
python ../../pyfunctor/script/csv_split.py ${DATA_DIR}/${DATA_FILENAME} ${SPLIT_PERCENT} "${DATA_DIR}/dev.csv" "${DATA_DIR}/train.csv"
