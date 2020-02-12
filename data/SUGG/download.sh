git clone https://github.com/Semeval2019Task9/Subtask-A.git

# prepare train set
cp Subtask-A/V1.4_Training.csv train.csv

# prepare dev set
sed '1d' Subtask-A/SubtaskA_Trial_Test_Labeled.csv > tmp.csv
iconv -f ISO-8859-1 -t UTF-8 tmp.csv > dev.csv

# clean
rm tmp.csv
rm -rf Subtask-A
