# Obtain FUNNY dataset
-Step 1: download Yelp Challenge Dataset 

    wget https://s3.amazonaws.com/stat.184.data/Yelp/yelp_academic_dataset_review.csv

-Step 2: run extraction code

    python extract.py

-Step 3: split the dataset

    sh split.sh 

# Reference
**[Identifying Humor in Reviews using Background Text Sources. (EMNLP 2017)](https://www.aclweb.org/anthology/D17-1051.pdf)**.
