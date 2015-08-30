#coding=utf-8

import unicodedata

class Tools(object) :
    @staticmethod
    def get_unichr_category(unichr_val) :
        '''
        logic : 
        unicode category : generaly , return 2 letters , 
                           like 'Lu' , 'L'(first char) standands for the char type , here means 'Letter' , 'u'(second char)stands for 'uppercase'
        so , we use the first letter as the type feature.
        '''
        category = unicodedata.category(unichr_val) #! return the unichr's category
    
    @staticmethod
    def is_unicode_number(unichr_val) :
        uni_type = Tools.get_unichr_category(unichr_val)
        return uni_type.startswith("N")

    @staticmethod
    def is_unicode_Lu_Ll_char(unichr_val) :
        uni_type = Tools.get_unichr_category(unichr_val)
        return uni_type == "Lu" or uni_type == "Ll"
    
    @staticmethod
    def get_unichr_type(unichr_val) :
        return Tools.get_unichr_category(unichr_val)
