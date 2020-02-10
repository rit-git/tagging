# Tagging
An experimental comparison of deep and simple models for semantic tagging

# Dependency
- pytorch
- pytorch-pretrained-bert
- sklearn

# Model
- BERT (Bidirectional Encoder representations from Transformers)
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- LR (Logistic Regression)
- SVM (Support Vector Machine)

# Run
-Step 1: clone the repository

    git clone --recursive https://github.com/rit-git/tagging.git

    cd tagging

-Step 2: install dependency

    pip install -r requirements.txt

-Step 2: evaluation

    cd script
    
    sh bert.sh

    cat result/bert.log

# Dataset

| Dataset | #Record    | %Positive   | Task       |
|---------|------------|-------------|------------|
| SUGG    | 9,092      | 0.261878575 | Suggestion |
| HOTEL   | 7,534      | 0.054021768 | Suggestion |
| SENT    | 11,379     | 0.097548115 | Tip        |
| PARA    | 6,566      | 0.167681998 | Tip        |
| FUNNY   | 4,871,770  | 0.02508616  | Humor      |
| HOMO    | 2,250      | 0.714222222 | Humor      |
| HETER   | 1,780      | 0.714044944 | Humor      |
| TV      | 13,447     | 0.525395999 | Spoiler    |
| BOOK    | 17,672,655 | 0.032237601 | Spoiler    |
| EVAL    | 10,386     | 0.383400732 | Argument   |
| REQ     | 10,386     | 0.183997689 | Argument   |
| FACT    | 10,386     | 0.364529174 | Argument   |
| REF     | 10,386     | 0.019930676 | Argument   |
| QUOTE   | 10,386     | 0.015501637 | Argument   |
| ARGUE   | 23,450     | 0.436972281 | Argument   |
| SUPPORT | 23,450     | 0.193944563 | Argument   |
| AGAINST | 23,450     | 0.243027719 | Argument   |
| AMAZON  | 3,600,000  | 0.5         | Sentiment  |
| YELP    | 560,000    | 0.5         | Sentiment  |
| FUNNY\* | 244,428    | 0.5         | Humor      |
| BOOK\*  | 1,139,448  | 0.5         | Spoiler    |

## Reference 
Deep or Simple models for Semantic Tagging? It Depends on your Data
