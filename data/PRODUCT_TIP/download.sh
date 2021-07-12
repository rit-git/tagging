#wget http://proj.ise.bgu.ac.il/public/gen_tips.zip
#unzip gen_tips.zip

python prepare.py

# split
SPLIT_PERCENT=0.2

python ../../pyfunctor/script/csv_split.py ./all.csv ${SPLIT_PERCENT} ./dev.csv ./train.csv
