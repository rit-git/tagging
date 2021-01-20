# 🔥 Semantic Tagging Benchmark
An experimental comparison of deep and simple models for semantic tagging 

21 distinctive datasets were used and they are available under folder data/

The datasets can also be used for broader NLP tasks including text/intent classification and information extraction

## Dataset

| Dataset | #Record | %Positive | Quality | Task       |
|---------|---------|-----------|---------|------------|
| SUGG    | 9K      | 0.26      | clean   | Tip        |
| HOTEL   | 8K      | 0.05      | clean   | Tip        |
| SENT    | 11k     | 0.10      | clean   | Tip        |
| PARA    | 7K      | 0.17      | clean   | Tip        |
| HOMO    | 2K      | 0.71      | clean   | Humor      |
| HETER   | 2K      | 0.71      | clean   | Humor      |
| FUNNY   | 5M      | 0.03      | dirty   | Humor      |
| FUNNY\* | 244K    | 0.50      | dirty   | Humor      |
| TV      | 13K     | 0.53      | clean   | Spoiler    |
| BOOK    | 18M     | 0.03      | dirty   | Spoiler    |
| BOOK\*  | 1M      | 0.50      | dirty   | Spoiler    |
| EVAL    | 10K     | 0.38      | clean   | Argument   |
| REQ     | 10K     | 0.18      | clean   | Argument   |
| FACT    | 10K     | 0.36      | clean   | Argument   |
| REF     | 10K     | 0.02      | clean   | Argument   |
| QUOTE   | 10K     | 0.02      | clean   | Argument   |
| ARGUE   | 23K     | 0.44      | clean   | Argument   |
| SUPPORT | 23K     | 0.19      | clean   | Argument   |
| AGAINST | 23K     | 0.24      | clean   | Argument   |
| AMAZON  | 4M      | 0.50      | clean   | Sentiment  |
| YELP    | 560K    | 0.50      | clean   | Sentiment  |

## Model
- BERT (Bidirectional Encoder representations from Transformers)
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- LR (Logistic Regression)
- SVM (Support Vector Machine)

## Appendix Model
- ALBERT (A Lite BERT)
- ROBERTA (A Robustly Optimized BERT Pretraining Approach)
- NB (Naive Bayes)
- XGboost (A Scalabel Tree Boosting System)

## Dependency
- pytorch 1.2.0
- pytorch-pretrained-bert 0.6.2
- scikit-learn 0.23.1
- transformers 2.3.0
- xgboost 1.1.0

## Run
-Step 1: clone the repository

    git clone --recursive https://github.com/rit-git/tagging.git

    cd tagging

-Step 2: install dependency

    pip install -r requirements.txt

-Step 3: prepare dataset 
    
    cd ./data/SUGG

    sh download.sh

    cd ../../

-Step 4: evaluate models

    cd script
    
    sh bert.sh

    cat result/bert.csv

-Step 5: More deep and simple models can be found under folders script/ and appendix/

## Citation
**[Deep or Simple models for Semantic Tagging? It Depends on your Data (PVLDB 2020 Sep., Tokyo)](http://www.vldb.org/pvldb/vol13/p2549-li.pdf)**

    @article{DBLP:journals/pvldb/LiL0T20,
        author    = {Jinfeng Li and
                     Yuliang Li and
                     Xiaolan Wang and
                     Wang{-}Chiew Tan},
        title     = {Deep or Simple Models for Semantic Tagging? It Depends on your Data},
        journal   = {Proc. {VLDB} Endow.},
        volume    = {13},
        number    = {11},
        pages     = {2549--2562},
        year      = {2020},
        url       = {http://www.vldb.org/pvldb/vol13/p2549-li.pdf},
        timestamp = {Tue, 24 Nov 2020 14:44:02 +0100},
        biburl    = {https://dblp.org/rec/journals/pvldb/LiL0T20.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
