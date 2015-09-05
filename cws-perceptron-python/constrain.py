#coding=utf-8

from symbols import TAG_B , TAG_M , TAG_E , TAG_S
from model import Model

class Constrain(object) :

    def __init__(self) :
        self.current_position_possible_labels = {}

    def clear(self) :
        self.current_position_possible_labels = {}
    
    def set_constrain_data(self , constrain_pos_list) :
        '''
        generate constrain data
        from [ 1, 4 , 6 , ..] separators position list 
        to   { [1] -> (E , S) , [2] -> (B , S) , [4] -> (E , S) , [5] -> (B , S) , ... }
        that is : for every [pos] in list , [pos] value is (E , S) , [pos+1] value is (B , S) , and this can construct a separator certainly .
        and by calling `get_current_position possible_labels` , we can ensure the partial segmentation structure can be kept in spite of the model.

        Args :
            constrain_pos_list : list , separator position list
        '''
        constrain_data = {}
        for pos in constrain_pos_list :
            constrain_data[pos] = ( TAG_E , TAG_S )
            constrain_data[pos+1] = ( TAG_B , TAG_S )
        self.current_position_possible_labels = constrain_data

    def get_current_position_possible_labels(self , pos) :
        if pos not in self.current_position_possible_labels :
            if pos == 0 :
                return ( TAG_B , TAG_S )
            else :
                return ( TAG_B , TAG_M , TAG_E , TAG_S )
        else :
            return self.current_position_possible_labels[pos]

    def get_possible_previous_label_at_current_label(self , pos , label) :
        if pos == 0 :
            return ( Model.BOS_LABEL_REP , )
        else :
            if label in (TAG_B , TAG_S) :
                return ( TAG_E , TAG_S )
            elif label in (TAG_M , TAG_E) :
                return ( TAG_B , TAG_M )
            else :
                logging.error("Invalid label : %s" %(label))
                logging.error("Exit!")
                exit(1)

