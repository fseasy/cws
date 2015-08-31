#coding=utf-8

from symbols import TAG_B , TAG_M , TAG_E , TAG_S
from models import Model.BOS_LABEL_REP as BOS_LABEL_REP
class Constrain(object) :

    def __init__(self) :
        self.current_position_possible_labels = {}

    def clear(self) :
        self.current_position_possible_labels = {}
    
    def set_constrain_data(self , constrain_data) :
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
            return ( BOS_LABEL_REP , )
        else :
            if label in (TAG_B , TAG_S) :
                return ( TAG_E , TAG_S )
            elif label in (TAG_M , TAG_E) :
                return ( TAG_B , TAG_M )
            else :
                logging.error("Invalid label : %s" %(label))
                logging.error("Exit!")
                exit(1)

