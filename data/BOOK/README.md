# Obtain BOOK dataset
-Step 1: download GoodReads review dataset

    download **[goodreads_reviews_spoiler.json.gz](https://drive.google.com/uc?id=196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS) from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews

-Step 2: uncompress

    gunzip goodreads_reviews_spoiler.json.gz

-Step 3: extract & split

    python extract.py

    sh split.sh

# Reference
**[Fine-Grained Spoiler Detection from Large-Scale Review Corpora (ACL 2019)](https://www.aclweb.org/anthology/P19-1248.pdf)**.
