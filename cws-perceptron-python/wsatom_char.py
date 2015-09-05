#coding=utf-8

import logging
from tools import Tools
from symbols import ENG_TYPE

class WSAtomTranslator(object) :
    @staticmethod
    def trans_unicode_list2atom_gram_list(unicode_list) :
        atom_list = []
        idx = 0
        unicode_num = len(unicode_list) 
        atom_unicode_list = []
        while idx < unicode_num :
            current_unicode = unicode_list[idx]
            atom_unicode_list.append(current_unicode)
            if not Tools.is_unicode_Lu_Ll_char(current_unicode) :
                type_str = ""
                if len(atom_unicode_list) > 1 :
                    type_str = ENG_TYPE
                else :
                    type_str = Tools.get_unichr_type(atom_unicode_list[0])
                atom_list.append(WSAtom(atom_unicode_list , type_str))
                atom_unicode_list = []
            idx += 1
        else :
            if len(atom_unicode_list) > 0 :
                # here it must be Lu or Ll Type char
                atom_list.append(WSAtom(atom_unicode_list , ENG_TYPE ))
                atom_unicode_list = []
        return atom_list
   
    @staticmethod
    def trans_atom_gram_list2unicode_list(atom_list) :
        unicode_list = []
        for atom in atom_list :
            unicode_list.append(unicode(atom))
        return unicode_list
    
    @staticmethod
    def trans_atom_gram_list2unicode_line(atom_list) :
        '''
        this module is a bit of confusion !!
        for the robust , do following logic .
        but it may be not clearly fundamentally .
        '''
        if type(atom_list) == list :
            unicode_list = WSAtomTranslator.trans_atom_gram_list2unicode_list(atom_list)
        elif isinstance(atom_list , WSAtom) :
            atom = atom_list
            unicode_list = [ unicode(atom) ]
        else :
            logging.error("Unkown Type for `atom list` ")
            unicode_list = [u"Unkonw Data Type"]
        return u"".join(unicode_list)


class WSAtom(object) :
    str_encoding="utf8"
    @classmethod
    def set_encoding(cls , encoding) :
        cls.str_encoding = encoding

    def __init__(self , unicode_list , type_str ) :
        self.unicode_data = unicode_list
        self.type = type_str
        self.unicode_str = u"".join(unicode_list)
        self.encoded_str = self.unicode_str.encode(WSAtom.str_encoding) 
    def get_type_str(self) :
        return self.type

    def __str__(self) :
        return self.encoded_str

    def __unicode__(self) :
        return self.unicode_str
