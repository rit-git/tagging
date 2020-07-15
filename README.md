# Tagging
An experimental comparison of deep and simple models for semantic tagging

# Dataset

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

# Model
- BERT (Bidirectional Encoder representations from Transformers)
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- LR (Logistic Regression)
- SVM (Support Vector Machine)

# Appendix Model
- ALBERT (A Lite BERT)
- ROBERTA (A Robustly Optimized BERT Pretraining Approach)
- NB (Naive Bayes)
- XGboost (A Scalabel Tree Boosting System)

# Dependency
- pytorch
- pytorch-pretrained-bert
- sklearn
- transformers
- xgboost

# Run
-Step 1: clone the repository

    git clone --recursive https://github.com/rit-git/tagging.git

    cd tagging

-Step 2: install dependency

    pip install -r requirements.txt

-Step 3: prepare dataset 
    
    cd ./data/SUGG

    sh download.sh

    cd ../../

-Step 4: evaluation

    cd script
    
    sh bert.sh

    cat result/bert.csv

# Reference 
**[Deep or Simple models for Semantic Tagging? It Depends on your Data](https://arxiv.org/abs/2007.05651)**
