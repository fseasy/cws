#coding=utf-8

import traceback
import logging
import sys

def get_file_encoding(f) :
    '''
    get file's encoding ; sample and naive implementation
    Args :
        f : file 
    Returns :
        encoding : str ;
    Attention :
        if failed , will cause to Exit !
    '''
    cur_g = f.tell()
    line = f.readline()
    f.seek(cur_g)
    encoding_list = []
    if len(f.encoding) > 0 :
        encoding_list.append(f.encoding)
    encoding_list.extend(["utf8","gb18030"])
    uline = ""
    for encoding in encoding_list :
        try :
            uline = line.decode(encoding)
        except :
            uline = ""
            continue
        return encoding
    raise UnicodeDecodeError , e :
        logging.error("failed to decode the training data")
        print >> sys.stderr , "Exit"
        exit(1)

def read_training_data(tf) :
    '''
    read lines from training dataset
    Args: 
        tf : file object of training data

    Returns :
        data_lines : lines of dataset 
    '''
    if type(tf) != file :
        try :
            tf = open(tf)
        except IOError , e :
            traceback.print_exc()
            exit(1)
    data_lines = []
    encoding = get_file_encoding(tf)
    for line in tf :
        uline = ""
        try :
            uline = line.decode(encoding)
        except :
            logging.warning("decoding dataset error : %s " %(line))
            continue
        data_lines.append(uline)
    logging.info("%d lines read done ." %(len(data_lines)))
    return data_lines