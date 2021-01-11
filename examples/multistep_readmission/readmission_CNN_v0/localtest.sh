mkdir output
mkdir vis
./CNN-trainvalidationtest.sh --input ../readmission_shared_data_preprocess_v0/output/ --output ./output/ --vis ./vis/ --trainratio 1.0 --validationratio 0.0 --testratio 0.0
