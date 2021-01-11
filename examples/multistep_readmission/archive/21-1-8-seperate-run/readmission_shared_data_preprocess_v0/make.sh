g++ -std=c++11 ./readmission_shared_data_preprocess.cpp -o ./bin/readmission_shared -lpthread -lgflags
g++ -std=c++11 ./readmission_shared_data_preprocess_inference.cpp -o ./bin/readmission_shared_inference -lpthread -lgflags

