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
from datasethandler import DatasetHandler

logging.config.fileConfig("logging.conf")

seg_logger = logging.getLogger('segmentor')
seg_logger.error("Test segmentor logger")

def seg_train(args) :
    if not DatasetHandler.is_readable(args.training_file) :
        logging.error("path '%s' open failed !" %(args.training_file))
        logging.error('Exit!')
        exit(1)
    if not DatasetHandler.is_readable( args.developing_file ) :
        logging.error("path '%s' open failed !" %(args.developing_file))
        logging.error("Exit!")
        exit(1)
    if not DatasetHandler.is_writeable(args.model_saving) :
        logging.error("path '%s' open failed !" %(args.model_saving))
        logging.error('Exit!')
        exit(1)
    segmentor = Segmentor()
    segmentor.train(args.training_file , args.developing_file , args.model_saving , args.max_iter )

def seg_predict(args) :
    if not DatasetHandler.is_readable(args.predict_file) :
        logging.error("path '%s' open failed !" %(args.predict_file))
        logging.error('Exit!')
        exit(1)
    if not DatasetHandler.is_readable( args.model_loading ) :
        logging.error("path '%s' open failed ! Model load Error ." %(args.model_loading))
        logging.error("Exit!")
        exit(1)
    if not DatasetHandler.is_writeable(args.output_path) and args.output_path != "stdout" :
        logging.error("path '%s' open failed !" %(args.output_path))
        logging.error('Exit!')
        exit(1)
    segmentor = Segmentor()
    segmentor.predict(args.predict_file , args.model_loading , args.output_path)

if __name__ == "__main__" :
    argp = argparse.ArgumentParser(description="averaged structured perceptron")
    sub_argps = argp.add_subparsers(title="model train" , description="From training dataset to train an averaged structured perceptron model")
    
    train_argp = sub_argps.add_parser("train")
    train_argp.add_argument("--training-file" , "-train" , help="training dataset for segmentation" , type=str, required=True)
    train_argp.add_argument("--developing-file" , "-dev" , help="developing dataset for segmentation" , type=str, required=True)
    train_argp.add_argument("--model-saving" , "-model" , help="training model saving path" , type=str, required=True)
    train_argp.add_argument("--max-iter" , "-ite" , help="training iteration time" , type=int)
    train_argp.set_defaults(func=seg_train)

    predict_argp = sub_argps.add_parser("predict")
    predict_argp.add_argument("--predict-file" , "-predict" , help="predict dataset for segmentation" , type=str , required=True)
    predict_argp.add_argument("--model-loading" , "-model" , help="model loading path" , type=str , required=True)
    predict_argp.add_argument("--output_path" , "-output" , help="output path[default : stdout]" , type=str , default="stdout") 
    predict_argp.set_defaults(func=seg_predict)

    args = argp.parse_args()
    args.func(args)

