#coding=utf-8

import traceback
import logging
import sys

from wsatom_char import WSAtomTranslator , WSAtom

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
            data_lines : lines of dataset , each line is also a list , every element is also a list !
                        the most inner element is WSAtom .
                        => [ [ [ "like" , "我" , ... ] , ["一" , "样"] ,...  ] ]
                        what is this ? -> the most inner list , is same as the N-grams of chars ! so as for every word , it is represented by
                                          a list of WSAtom . upper list is the sentence , the most outer is the list of sentence
                        Why use WSAtom ? -> because we want a English word as a `single representation` instead of `list of letters` ! 
        '''
        logging.info("reading training data .")
        if type(tf) != file :
            try :
                tf = open(tf)
            except IOError , e :
                traceback.print_exc()
                exit(1)
        data_lines = []
        encoding = DatasetHandler.get_file_encoding(tf)
        WSAtom.set_encoding(encoding)
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
            atom_list = []
            for uline_part in uline_parts :
                atom_list.append(WSAtomTranslator.trans_unicode_list2atom_gram_list(uline_part))
            data_lines.append(atom_list)
        logging.info("%d lines read done ." %(len(data_lines)))
        return data_lines
