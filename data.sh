wget https://raw.githubusercontent.com/ThunderingII/nlp_ner/master/data/data_train.txt
wget https://raw.githubusercontent.com/ThunderingII/nlp_ner/master/data/data_dev.txt
wget https://raw.githubusercontent.com/ThunderingII/nlp_ner/master/data/data_test.txt

mkdir -p data
mv data_*.txt data