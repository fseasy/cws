#/bin/sh
python main.py train -train /users1/wxu/download/doingcws/cws-perceptron-python/sampledata/debug.txt -dev /users1/wxu/download/doingcws/cws-perceptron-python/sampledata/lite_dev.txt  -model debugmodel.pkl.gz -ite 1 

#python main.py predict -predict /users1/wxu/download/doingcws/cws-perceptron-python/sampledata/lite_training.txt -model model.pkl.gz -output tmp

#python main.py eval -gold /users1/wxu/download/doingcws/cws-perceptron-python/sampledata/lite_training.txt -predict tmp
