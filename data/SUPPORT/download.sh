# Obtain argument dataset
wget https://www.informatik.tu-darmstadt.de/media/ukp/data/fileupload_2/argument_annotated_news_articles/UKP_Sentential_Argument_Mining_Corpus.zip


unzip -o UKP_Sentential_Argument_Mining_Corpus.zip 

mv UKP\ Sentential\ Argument\ Mining\ Corpus dataset

data_dir="dataset"
# run the completion pipeline as required by the authors

cd ${data_dir}/completion-pipeline

sed '125d' src/main/java/de/tudarmstadt/ukp/dkpro/argumentation/sentential/completion/CompletionPipeline.java > tmp.java

mv tmp.java src/main/java/de/tudarmstadt/ukp/dkpro/argumentation/sentential/completion/CompletionPipeline.java

mvn compile

mvn exec:java -Dexec.args="-i ../data/incomplete -o ../data/complete"

cd ../../

python extract_dataset.py

python extract.py Argument_for

# clean
rm -rf UKP_Sentential_Argument_Mining_Corpus.zip
rm -rf ${data_dir}
