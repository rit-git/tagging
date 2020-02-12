# Tagging
An experimental comparison of deep and simple models for semantic tagging

# Dataset

| Dataset | #Record | %Positive | Quality | Task       |
|---------|---------|-----------|---------|------------|
| SUGG    | 9K      | 0.26      | clean   | Suggestion |
| HOTEL   | 8K      | 0.05      | clean   | Suggestion |
| SENT    | 11k     | 0.10      | clean   | Tip        |
| PARA    | 7K      | 0.17      | clean   | Tip        |
| HOMO    | 2K      | 0.71      | clean   | Humor      |
| HETER   | 2K      | 0.71      | clean   | Humor      |
| TV      | 13K     | 0.53      | clean   | Humor      |
| EVAL    | 10K     | 0.38      | clean   | Spoiler    |
| REQ     | 10K     | 0.18      | clean   | Spoiler    |
| FACT    | 10K     | 0.36      | clean   | Argument   |
| REF     | 10K     | 0.02      | clean   | Argument   |
| QUOTE   | 10K     | 0.02      | clean   | Argument   |
| ARGUE   | 23K     | 0.44      | clean   | Argument   |
| SUPPORT | 23K     | 0.19      | clean   | Argument   |
| AGAINST | 23K     | 0.24      | clean   | Argument   |
| FUNNY   | 5M      | 0.03      | dirty   | Argument   |
| BOOK    | 18M     | 0.03      | dirty   | Argument   |
| AMAZON  | 4M      | 0.50      | clean   | Sentiment  |
| YELP    | 560K    | 0.50      | clean   | Sentiment  |
| FUNNY\* | 244K    | 0.50      | dirty   | Humor      |
| BOOK\*  | 1M      | 0.50      | dirty   | Spoiler    |

# Model
- BERT (Bidirectional Encoder representations from Transformers)
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- LR (Logistic Regression)
- SVM (Support Vector Machine)

# Dependency
- pytorch
- pytorch-pretrained-bert
- sklearn

# Run
-Step 1: clone the repository

    git clone --recursive https://github.com/rit-git/tagging.git

    cd tagging

-Step 2: install dependency

    pip install -r requirements.txt

-Step 3: evaluation

    cd script
    
    sh bert.sh

    cat result/bert.csv

# Reference 
Deep or Simple models for Semantic Tagging? It Depends on your Data
