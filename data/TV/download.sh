# Obtain PARA dataset
wget https://www.cs.colorado.edu/~jbg/downloads/spoilers.tar.gz

mkdir spoilers

tar zxvf spoilers.tar.gz -C spoilers

python extract_tv.py 

# clean
rm -rf spoilers.tar.gz 
rm -rf spoilers
