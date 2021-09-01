wget https://helpful-sentences-from-reviews.s3.amazonaws.com/train.json
wget https://helpful-sentences-from-reviews.s3.amazonaws.com/test.json

python ./process.py ./train.json ./train.csv 
python ./process.py ./test.json ./dev.csv
