git clone https://github.com/adobe-research/deft_corpus.git
mkdir train
python deft_corpus/task1_converter.py deft_corpus/data/deft_files/train ./train
python def_convert.py train

mkdir dev
python deft_corpus/task1_converter.py deft_corpus/data/deft_files/dev ./dev
python def_convert.py dev

