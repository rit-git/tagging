# Obtain PARA dataset
wget http://xinyuhua.github.io/resources/naacl2019/naacl19_dataset.zip

unzip -o naacl19_dataset.zip

python extract_naacl19_dataset.py

python extract.py evaluation

# clean
rm -rf naacl19_dataset.zip
rm -rf dataset
