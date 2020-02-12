# Obtain YELP dataset
-Step 1: download yelp polarity classification dataset

    download yelp_review_polarity_csv.tar.gz from http://goo.gl/JyCnZq 

-Step 2: uncompress

    tar zxvf yelp_review_polarity_csv.tar.gz

-Step 3: extract

    python transform.py yelp_review_polarity_csv/train.csv train.csv

    python transform.py yelp_review_polarity_csv/test.csv test.csv

# Reference
**[Character-level Convolutional Networks for Text Classification (NIPS 2016)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)**.
