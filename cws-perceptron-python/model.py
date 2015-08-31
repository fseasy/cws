#coding=utf-8

from symbols import DEBUG

class Model(object) :
    BOS_LABEL_REP = None
    def __init__(self) :
        self.emit_feature_space = None #! feature space and feature 2 feature idx translator. dict
        self.label_space = None #! label space and label to label idx translator . dict
        self.idx2emit_feature = None #! reverse index for emit feature . ONLY DEBUG NEEDED
        self.idx2label = None #! reverse index for label

        self.emit_feature_num = 0
        self.emit_feature_trans = None #! so called trans , is a 2-d array . storing the index of the joint (feature_idx , label_idx) for X vector
        self.label_num = 0
        self.trans_feature_trans = None #! same as emit_feature trans , storing the idx of (from_label , to_label) for X vector
        
        self.W = None   #! current weight vector
        self.W_sum = None  #! for average
        self.W_time = None #! record timestamp
        self.W_size = 0
        self.time_now = 0
        
        self.emit_feature_cache = None #! it is always one same emit feature and many different label , 
                                       #! so we can make a cache to decrease time cost . But , it may count less .
        self.emit_feature_cached_idx = None 

    def init_empty_model(self) :
        self.emit_feature_space = {}
        self.label_space = {}
        if DEBUG :
            self.idx2emit_feature = []
        self.idx2label = []

    def add_emit_feature_list2feature_space(self , features_list ) :
    '''
    add the emit feature to feature space
    Args :
        features_list : list of list . most inner element is emit feature , a str like "1=我" 
                        => [ ["1=我" , "2=BOS" , ...] , ...  ]
    '''
        for features in features_list :
            for emit_feature in features :
                if emit_feature not in self.emit_feature_space :
                    self.emit_feature_space[emit_feature] = self.emit_feature_num # str -> idx
                    self.emit_feature_num += 1
                    if DEBUG :
                        self.idx2emit_feature.append(emit_feature)
                # if after we need to cut the model , we may need to record the hit num of a specific feature

    def add_labels2label_space(self , labels) :
        for label in labels :
            self.label_space[label] = self.label_num
            self.label_num += 1
            self.idx2label.append(label)

    def build_up_model(self) :
        self._build_emit_feature_trans()
        self._build_trans_feature_trans()
        self._build_empty_weight_and_relevant()


    def _build_emit_feature_trans(self) :
        '''
        trans : (emit_feature/idx , label/idx ) -> instance feature idx (corresponding weight feature idx)
        we calculate and store it for decrease indexing cost
        the structure is this :
            |--- feature1 at every label ---|--- feature2 at every label ---| --- .. . ---|--- featureN at every label ---| 
        so :
            feature_idx * label_num + label_idx => instance idx(emit feature)
        '''
        logging.info("build emit feature translation struct.")
        if self.emit_feature_space is None :
            logging.warn("failed")
            return
        self.emit_feature_trans = [ [ 0 ] * self.label_num for n in range(self.emit_feature_num) ]
        for f_idx in range(self.emit_feature_num) :
            for l_idx in range(self.label_num) :
                self.emit_feature_trans[f_idx][l_idx] = f_idx * self.label_num + l_idx

    def _build_trans_feature_trans(self) :
        '''
        thoughts is same as emit feature trans
        Attention : a extend label should be added ! ==> the label of BOS , LBOS
                    for the first label , it's previous label also should be added .
                    for LBOS , because it is special and is only used when transport from the head , 
                    so we are just using `None` to tag it when need !!
        structure :
            |--- label0 to every label ---|--- ... ---|---labelM to every label ---|--- LBOS to every_label ---|
        and correspoding to the weight idx , we should add the offset of emit feature . which value is emit_feature_num * label_num .
        so for(pre_label/idx , to_label/idx ) :
            emit_feature_num * label_num + pre_label_idx * label_num + to_label_idx => instance idx(trans feature)
        for LBOS :
            emit_feature_num * label_num + label_num * label_num + to_label_idx => instance idx(trans feature)
        '''
        logging.info("build transition feature translation struct . ")
        if self.label_space is None or self.emit_feature_space is None :
            logging.info("failed")
            return
        self.trans_feature_trans = [ [ 0 ] * self.label_num for n in range(self.label_num + 1)]
        offset = self.emit_feature_num * self.label_num 
        for pre_label_idx in range(self.label_mum) :
            for to_label_idx in range(self.label_num) :
                self.trans_feature_trans[pre_label_idx][to_label_idx] = offset + pre_label_idx * self.label_num + to_label_idx
        #! calculate the LBOS idx
        offset = offset + self.label_num * self.label_num 
        for to_label_idx in range(self.label_num) :
            self.trans_feature_trans[self.label_num][to_label_idx] = offset + to_label_idx
        
        
    def _build_empty_weight_and_relevant(self) :
        '''
        init empty weight vector
        '''
        logging.info("build weight and relevat struct .")
        if self.label_space is None or self.emit_feature_space is None :
            logging.error("please build the label space and emit feature space before initialize weight vector")
            logging.error("Exit")
            exit(1)
        #! W = ( emit features for every label ) + ( every label to every label ) + ( LBOS to every label ) 
        self.W_size = self.label_num * self.label_num + self.emit_feature_num * self.label_num + self.label_num
        self.W = {} #! as sparse vector . A class may be more Nice . but dict is also enough
        self.W_sum = {} 
        self.W_time = {} #! defalt vlaue is the init self.time_now  

    def increase_time(self) :
        self.time_now += 1

    def _get_emit_feature_idx(self , emit_feature) :
        '''
        Trans emit feature(list of feature str) to emit feature idx (list of feature idx )
        '''
        feature_idx = []
        if emit_feature == self.emit_feature_cache :
            feature_idx = self.emit_feature_cached_idx
        else :
            for f in emit_feature :
                if f in self.emit_feature_space :
                    feature_idx.append(self.emit_feature_space[f])
            self.emit_feature_cache = emit_feature
            self.emit_feature_cached_idx = emit_feature_idx
        return feature_idx
    
    def _get_instance_vector_idx_list_for_emit_feature(self , emit_feature_idx , label_idx) :
        '''
        base on  emit feature idx and label idx , return the corresponding instance idx (list)
        transition rules are befined in `_build_emit_feature_trans` , here just using the cached result mat
        Returns : 
            list of idx
        '''
        f = []
        for idx in emit_feature_idx :
            f.append(self.emit_feature_trans[idx][label_idx])
        return f

    def _get_instance_vector_idx_for_trans_feature(self , pre_label_idx , current_label_idx) :
        if pre_label_idx == Model.BOS_LABEL_REP :
            return self.trans_feature_trans[self.label_num][current_label_idx]
        else :
            return self.trans_feature_trans[pre_label_idx][current_label_idx]

    def build_instance_vector_for_one_term(self , emit_feature , pre_label , current_label) :
        '''
        build one term feature vector
        we haven't check the idx ! it may cause Error . but we ignore it currently .
        '''
        instance_vector = self.get_instance_emit_vector_for_one_term(self , emit_feature , current_label )
        instance_vector.update(self.get_instance_trans_vector_for_one_term(self , pre_label , current_label))
        return instance_vector 

    def phi(self , X , pre_y , y )
    '''
    Just for a nice API
    '''
        return self.build_instance_vector_for_one_term(X , pre_y , y)

    def get_instance_emit_vector_for_one_term(self , emit_feature , label) :
        emit_feature_idx = self._get_emit_feature_idx(emit_feature)
        label_idx = self.trans_label2idx(label)
        instance_emit_vector_idx = self._get_instance_vector_idx_list_for_emit_feature(emit_feature_idx , label_idx)
        return dict.fromkeys(instance_emit_vector_idx , 1) #! bool feature

    def get_instance_trans_vector_for_one_term(self , pre_label , current_label) :
        pre_label_idx = self.trans_label2idx(pre_label) 
        current_label_idx = self.trans_label2idx(current_label)
        instance_vector_trans_idx = self._get_instance_vector_idx_for_trans_feature(pre_label_idx , current_label_idx)
        return { instance_vector_trans_idx : 1 } #! instance_vector_trans_idx is a number

    def build_instance_vector_by_combining_emit_and_trans(self , emit_vector , trans_vector) :
        '''
        for efficient , we will modified emit_vector !!
        '''
        
        assert(isinstance(emit_vector , dict) and isinstance(trans_vector , dict))
        emit_vector.udpate(trans_vector) 
        return emit_vector

    def get_label_num(self) :
        return self.label_num

    def get_current_weight_vector(self) :
        return self.W 
            emit_feature = extractor.
    
    def trans_label2idx(self , label) :
        if label == Model.BOS_LABEL_REP 
            return Model.BOS_LABEL_REP 
        else :
            return self.label_space[label]

    def trans_idx2label(self , label_idx) :
        return self.idx2label[label_idx]

    def trans_iex2emit_feature(self , emit_feature_idx) :
        if DEBUG :
            return self.idx2emit_feature[emit_feature_idx]
        else :
            logging.warning("Attempt to access self.idx2emit_feature . But DEBUG mode is closed !")
            logging.warning("'None' is returned")
            return "None"
