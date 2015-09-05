#coding=utf-8

import traceback
import logging
import os
import sys

from wsatom_char import WSAtomTranslator , WSAtom

class DatasetHandler(object) : 
    @staticmethod
    def is_readable(path) :
        return ( os.access(path , os.F_OK) and os.access(path , os.R_OK) )

    @staticmethod
    def is_writeable(path) :
        if os.access(path , os.F_OK) :
            return os.access(path , os.W_OK)
        #! path not exists , check dir path is writeable
        dir_path = os.path.dirname(os.path.abspath(path)) #!! os.path.abspath is needed !
                                                          #~  or an empty str is returned for dirname for a relative path
        return ( os.access(dir_path , os.F_OK) and os.access(dir_path , os.W_OK) )

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
        logging.error("failed to decode the training data . file path : '%s'" %(f.name))
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
        tf.close()
        return data_lines
    
    @staticmethod
    def read_dev_data(df) :
        '''
        An Iteration generator for developing data
        Args :
            df : file , or a path str
        Returns :
            atom_list : [ [WSatom , ....] , ...  ] 
        '''
        if not isinstance(df , file) :
            try :
                df = open(df)
            except IOError , e :
                traceback.print_exc()
                exit(1)
        encoding = DatasetHandler.get_file_encoding(df)
        WSAtom.set_encoding(encoding)
        for line in df :
            line = line.strip()
            if len(line) == 0 :
                continue
            try :
                uline = line.decode(encoding)
            except UnicodeDecodeError , e :
                logging.warning("decoding dataset error : %s " %(line))
                continue
            uline_parts = uline.split()
            atom_list = []
            for uline_part in uline_parts :
                atom_list.append(WSAtomTranslator.trans_unicode_list2atom_gram_list(uline_part))
            yield atom_list
        df.close()

    @staticmethod
    def read_predict_data(df) :
        '''
        An Iteration generator for predict data
        Args :
            df : file , or a path str
        Returns :
            atom_list : [ WSAtom , WSAtom , ... ] 
            separator_position : list , the position where seperator exists
        '''
        if not isinstance(df , file) :
            try :
                df = open(df)
            except IOError , e :
                traceback.print_exc()
                exit(1)
        encoding = DatasetHandler.get_file_encoding(df)
        WSAtom.set_encoding(encoding)
        for line in df :
            line = line.strip()
            #if len(line) == 0 : #!! we can't continue any more ! keep a empty line is very necessary !!!
            #    continue
            try :
                uline = line.decode(encoding)
            except UnicodeDecodeError , e :
                logging.warning("decoding dataset error : %s " %(line))
                #continue #! keep it !
                uline = ""
            uline_parts = uline.split()
            atom_list = []
            for uline_part in uline_parts :
                atom_list.append(WSAtomTranslator.trans_unicode_list2atom_gram_list(uline_part)) 
            #! => atom_list : [ [ WSAtom , WSAtom , ... ] , [...] , ... ]
            #~ we should record the seperator position , and put all inner WSAtom in different inner list to a full list .
            #~ the seperator position is the place between 2 inner list , we set the value as the len(pre_WSAtom) -1 . that means :
            #~ the origin position of separator is after the WSAtom index in the full list 
            #~ how fucking is the explanation . code may be more clearly !!
            atom_unigrams = []
            separator_pos = []
            pos = 0
            part_num = len(atom_list)
            for idx in range(atom_list) :
                atoms = atom_list[idx]
                pos += len(atoms)
                atom_unigrams.extend(atoms)
                if idx != 0 and idx != part_num -1 :
                    separator_pos.append(pos -1 ) #!! pos -1
            yield atom_unigrams , separator_pos 
        df.close()
