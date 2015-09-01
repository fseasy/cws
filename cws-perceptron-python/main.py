#/usr/bin/env python
#coding=utf-8
'''
# * Author        : xu wei
# * Email         : readonlyfile@hotmail.com
# * Create time   : 2015-08-29 18:08
# * Last modified : 2015-08-29 18:08
# * Filename      : main.py
# * Description   : 
'''
import argparse
import logging
import logging.config
from segmentor import Segmentor

logging.config.fileConfig("logging.conf")

seg_logger = logging.getLogger('segmentor')
seg_logger.error("Test segmentor logger")

def main(training_f , model_saving_f) :
    segmentor = Segmentor()
    segmentor.train(training_f , model_saving_f)

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="averaged structed perceptron")
    argp.add_argument("--training-file" , "-train" , help="training data set for segmentation" , type=argparse.FileType('r') , required=True)
    argp.add_argument("--model-saving" , "-save" , help="training model saving path" , type=argparse.FileType('w') , required=True)
    args = argp.parse_args()
    
    main(args.training_file , args.model_saving)

    args.training_file.close()
    args.model_saving.close()
