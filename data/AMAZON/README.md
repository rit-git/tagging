# Obtain AMAZON dataset
-Step 1: download amazon polarity classification dataset

    download amazon_review_polarity_csv.tar.gz from https://drive.google.com/drive/u/1/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

-Step 2: uncompress

    tar zxvf amazon_review_polarity_csv.tar.gz

-Step 3: extract

    python transform.py amazon_review_polarity_csv/train.csv train.csv

    python transform.py amazon_review_polarity_csv/test.csv test.csv

# Reference
**[Character-level Convolutional Networks for Text Classification (NIPS 2016)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)**.
