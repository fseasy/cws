#/bin/sh
python main.py train -train /users1/wxu/download/doingcws/cws-perceptron-python/sampledata/lite_training.txt -dev /users1/wxu/download/doingcws/cws-perceptron-python/sampledata/lite_training.txt  -model model.pkl.gz 

python main.py predict -predict /users1/wxu/data/cws_corpus/test.txt -model model.pkl.gz -output stdout
