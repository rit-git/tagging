# Obtain PARA dataset
wget http://alt.qcri.org/semeval2017/task7/data/uploads/semeval2017_task7.tar.xz

tar xf semeval2017_task7.tar.xz

python extract_heter.py 

# clean
rm -rf semeval2017_task7.tar.xz
rm -rf semeval2017_task7
