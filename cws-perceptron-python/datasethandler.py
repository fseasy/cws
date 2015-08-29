#coding=utf-8

import traceback
import logging
import sys

class DatasetHandler(object) : 
    @staticmethod
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
        if f.encoding is not None :
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
        logging.error("failed to decode the training data")
        print >> sys.stderr , "Exit"
        exit(1)
    
    @staticmethod
    def read_training_data(tf) :
        '''
        read lines from training dataset
        Args: 
            tf : file object of training data
    
        Returns :
            data_lines : lines of dataset , each line is also a list , every element is a word 
        '''
        if type(tf) != file :
            try :
                tf = open(tf)
            except IOError , e :
                traceback.print_exc()
                exit(1)
        data_lines = []
        encoding = DatasetHandler.get_file_encoding(tf)
        for line in tf :
            line = line.strip()
            if len(line) == 0 : 
                continue
            uline = ""
            try :
                uline = line.decode(encoding)
            except :
                logging.warning("decoding dataset error : %s " %(line))
                continue
            uline_parts = uline.split()
            data_lines.append(uline_parts)
        logging.info("%d lines read done ." %(len(data_lines)))
        return data_lines
