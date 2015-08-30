#coding=utf-8

class Model(object) :
    def __init__(self) :
        self.emit_feature_space = None #! feature space and feature 2 feature idx translator. dict
        self.label_space = None #! label space and label to label idx translator . dict
        
        self.emit_feature_num = 0
        self.emit_feature_trans = None #! so called trans , is a 2-d array . storing the index of the joint (feature_idx , label_idx) for X vector
        self.label_num = 0
        self.trans_feature_trans = None #! same as emit_feature trans , storing the idx of (from_label , to_label) for X vector
        
        self.W = None   #! current weight vector
        self.W_sum = None  #! for average
        self.W_time = None #! record timestamp




