#coding=utf-8

import unicodedata

class Tools(object) :
    @staticmethod
    def get_unichr_type(unichr_val) :
        '''
        logic : 
        unicode category : generaly , return 2 letters , 
                           like 'Lu' , 'L'(first char) standands for the char type , here means 'Letter' , 'u'(second char)stands for 'uppercase'
        so , we use the first letter as the type feature.
        '''
        category = unicodedata.category(unichr_val) #! return the unichr's category
