dataset="SUGG"
cp output/${dataset}_zip.csv ../calibrate/output.csv
cd ../calibrate
sh calibrate.sh
cd ../calibrate_svm
