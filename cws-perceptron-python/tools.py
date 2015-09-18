#coding=utf-8

import unicodedata
from collections import Counter
class Tools(object) :
    @staticmethod
    def get_unichr_category(unichr_val) :
        '''
        logic : 
        unicode category : generaly , return 2 letters , 
                           like 'Lu' , 'L'(first char) standands for the char type , here means 'Letter' , 'u'(second char)stands for 'uppercase'
        so , we use the first letter as the type feature.
        '''
        return unicodedata.category(unichr_val) #! return the unichr's category
    
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

    @staticmethod
    def sparse_vector_dot(v1 , v2) :
        assert(isinstance(v1 , dict) and isinstance(v2 , dict))
        l1 = len(v1)
        l2 = len(v2)
        if l1 < l2 :
            small_v = v1 
            big_v = v2
        else :
            small_v = v2
            big_v = v1
        rst = 0
        for k in small_v :
            if k not in big_v :
                continue
            val_1 = small_v[k]
            val_2 = big_v[k]
            rst += val_1 * val_2
        return rst

    @staticmethod
    def sparse_vector_add(*args) :
        rst = Counter()
        for v in args :
            assert(isinstance(v , dict))
            rst.update(v)
        return Tools.clear_zero_value(dict(rst))
    
    @staticmethod
    def spase_vector_add_in_place(augend_v , addend_v) :
        keys = list(set(augend_v.keys() + addend_v.keys()))
        for k in keys :
            k_in_augend = k in augend_v
            k_in_addend = k in addend_v
            if k_in_augend and k_in_addend :
                added_value = augend_v[k] + addend_v[k]
                if added_value == 0 : 
                    del augend_v[k]
                else :
                    augend_v[k] = added_value
            elif (not k_in_augend)  and k_in_addend :
                augend_v[k] = addend_v[k]
    
    @staticmethod
    def sparse_vector_sub( minuend_v , subtrahend_v) :
        rst = Counter(minuend_v)
        rst.subtract(subtrahend_v)
        return Tools.clear_zero_value(dict(rst))

    @staticmethod
    def clear_zero_value(d) :
        keys = d.keys()
        for k in keys :
            if d[k] == 0 :
                del d[k]
        return d
