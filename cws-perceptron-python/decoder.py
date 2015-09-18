#coding=utf-8
from symbols import NEG_INF 
from tools import Tools

class Decoder(object) :
    def __init__(self) :
        self.instance_cached = None 
        self.emit_feature_list_cached = None
        self.instance_vector_mat_cached = None
        self.back_path_mat_cached = None
        self.current_instance_vector_list = None

    def decode(self , extractor , model , constrain ,instance) :
        '''
        Vertebi Decode . Attention , this decode is for Training . so cache and instance vector mat is set .
        Args :
            model : Model
            instance : list , list of WSAtom(unigram)
        '''
        instance_len = len(instance)
        label_num = model.get_label_num()
        #! score mat is structed like :
        #~ pos0_l0      pos1_l0     pos2_l0     ...Position incease... posL-1_l0
        #~ .
        #~ Label increase           # shape = ( label_num x instance_length ) #
        #~ .
        #~ pos0_lN-1    pos1_lN-1   pos2_lN-1   ... posL-1_LN-1
        score_mat = [ [ NEG_INF ] * instance_len for n in range(label_num) ] 
        #! back path mat is structued corresponding to score mat
        back_path_mat = [ [ None ] * instance_len for n in range(label_num) ]
        #! instance vector mat . cached it for update .
        instance_vector_mat = [ [ None ] * instance_len for n in range(label_num)]
        #! for one instance , w keep unchange . and for train , w is the current w , that is , the latest weight
        w = model.get_current_weight_vector()
        emit_feature_list = extractor.extract_emit_features(instance)
        for pos in range(instance_len) :
            emit_feature = emit_feature_list[pos]
            possible_labels = constrain.get_current_position_possible_labels(pos)
            for cur_label in possible_labels :
                previous_possible_labels = constrain.get_possible_previous_label_at_current_label(pos , cur_label)
                cur_label_idx = model.trans_label2idx(cur_label)
                max_trans_score = NEG_INF
                max_trans_score_cor_label_idx = None
                for pre_label in previous_possible_labels :
                    trans_vector = model.get_instance_trans_vector_for_one_term(pre_label , cur_label)
                    trans_score = Tools.sparse_vector_dot(trans_vector , w )
                    pre_label_idx = model.trans_label2idx(pre_label)
                    pre_label_score = score_mat[pre_label_idx][pos-1] if pos -1 >= 0 else 0
                    tmp_score = trans_score + pre_label_score
                    if tmp_score > max_trans_score :
                        max_trans_score = tmp_score
                        max_trans_score_cor_label_idx = pre_label_idx
                # calculate emit score
                emit_vector = model.get_instance_emit_vector_for_one_term(emit_feature , cur_label)
                emit_score = Tools.sparse_vector_dot(emit_vector , w )
                # record
                score_mat[cur_label_idx][pos] = max_trans_score + emit_score #! Atention 
                                                                             #~ max trans score is also including the previos label score
                back_path_mat[cur_label_idx][pos] = max_trans_score_cor_label_idx
                instance_vector_mat[cur_label_idx][pos] = model.build_instance_vector_by_combining_emit_and_trans(emit_vector , trans_vector)
        #! find the label of the end pos which has the max score
        max_score = NEG_INF
        max_score_cor_label_idx = None
        for l_idx in range(label_num) :
            score = score_mat[l_idx][instance_len-1]
            if score > max_score :
                max_score = score
                max_score_cor_label_idx = l_idx
        #! build the revered predict tag list and coresponding instance vector list
        predict_tags_list_r = [model.trans_idx2label(max_score_cor_label_idx)]
        instance_vector_list_r = [ instance_vector_mat[max_score_cor_label_idx][instance_len -1] ]
        for i in range(instance_len -1 , 0 , -1) : #!! ( len - 1 ) -> 1
            max_score_cor_label_idx = back_path_mat[max_score_cor_label_idx][i]
            #print "pos %d , char %s , label %s" %(i , instance[i] , max_score_cor_label_idx)
            predict_tags_list_r.append(model.trans_idx2label(max_score_cor_label_idx))
            instance_vector_list_r.append(instance_vector_mat[max_score_cor_label_idx][i-1]) #!! i - 1
        predict_tags_list_r.reverse() #! reverse , in replace !
        instance_vector_list_r.reverse()
        #! set cache
        self.instance_cached = instance
        self.emit_feature_list_cached = emit_feature_list
        self.instance_vector_mat_cached = instance_vector_mat 
        self.back_path_mat_cached = back_path_mat
        self.current_instance_vector_list = instance_vector_list_r 
        
        return predict_tags_list_r #! in fact , it is not reverse any more

    def get_current_predict_label_sequence_feature(self) :
        return Tools.sparse_vector_add( *self.current_instance_vector_list ) 

    def calculate_label_sequence_feature(self , label_sequence ,instance , extractor , model ) :
        if instance == self.instance_cached :
            emit_feature_list = self.emit_feature_list_cached
            instance_vector_mat = self.instance_vector_mat_cached 
            back_path_mat = self.back_path_mat_cached
        else :
            emit_feature_list = extracot.extract_emit_features(instance)
            instance_vector_mat = None
            back_path_mat = None
        instance_vector_list = []
        for pos in range(len(label_sequence)) :
            current_label = label_sequence[pos]
            pre_label = label_sequence[pos-1] if pos-1 >= 0 else model.BOS_LABEL_REP
            cached_v = self._try_to_get_one_term_vector_from_cached_data(instance_vector_mat , back_path_mat ,pos , pre_label , current_label , model)
            if cached_v is None :
                emit_feature = emit_feature_list[pos]
                calced_v = model.phi(emit_feature , pre_label , current_label)
                instance_vector_list.append(calced_v)
            else :
                instance_vector_list.append(cached_v)
        return Tools.sparse_vector_add( *instance_vector_list )


    def _try_to_get_one_term_vector_from_cached_data(self , instance_vector_mat , back_path_mat , pos , pre_label , current_label , model) :
        if instance_vector_mat is None or back_path_mat is None :
            return None
        pre_label_idx = model.trans_label2idx(pre_label)
        current_label_idx = model.trans_label2idx(current_label)
        if back_path_mat[current_label_idx][pos] != pre_label_idx :
            return None     #! trans path is not same
        return instance_vector_mat[current_label_idx][pos] #! from imagination , it also may None ! but return None is also in control
    
    @staticmethod 
    def decode_for_predict(extractor , model , constrain ,instance) :
        '''
        Vertebi Decode . this is for predict , the different between `decode` is :
            1. using average weight
            2. no cache
        this may be  very ##redundent## because kernel code  is the same to `decode` . 
        In fact , I copied and make a little change for this . 
        Just add a flag in `decode` can replace this function . 
        Why I chose this way ? may be i don't want to change previous code . what's more , too much flag is also a bad impact for code reading !
        Args :
            model : Model
            instance : list , list of WSAtom(unigram)
        '''
        instance_len = len(instance)
        label_num = model.get_label_num()
        #! score mat is structed like `decode`  :
        score_mat = [ [ NEG_INF ] * instance_len for n in range(label_num) ] 
        back_path_mat = [ [ None ] * instance_len for n in range(label_num) ]
        #! for predict , w is the average weight .
        w = model.get_average_weight_vevtor()
        
        emit_feature_list = extractor.extract_emit_features(instance)
        for pos in range(instance_len) :
            emit_feature = emit_feature_list[pos]
            #!! DEBUGING
            #f.write(str(emit_feature))
            #f.write("\n")

            possible_labels = constrain.get_current_position_possible_labels(pos)
            for cur_label in possible_labels :
                previous_possible_labels = constrain.get_possible_previous_label_at_current_label(pos , cur_label)
                cur_label_idx = model.trans_label2idx(cur_label)
                max_trans_score = NEG_INF
                max_trans_score_cor_label_idx = None
                for pre_label in previous_possible_labels :
                    trans_vector = model.get_instance_trans_vector_for_one_term(pre_label , cur_label)
                    trans_score = Tools.sparse_vector_dot(trans_vector , w )
                    pre_label_idx = model.trans_label2idx(pre_label)
                    pre_label_score = score_mat[pre_label_idx][pos-1] if pos -1 >= 0 else 0
                    tmp_score = trans_score + pre_label_score
                    if tmp_score > max_trans_score :
                        max_trans_score = tmp_score
                        max_trans_score_cor_label_idx = pre_label_idx
                # calculate emit score
                emit_vector = model.get_instance_emit_vector_for_one_term(emit_feature , cur_label)

                emit_score = Tools.sparse_vector_dot(emit_vector , w )
                # record
                score_mat[cur_label_idx][pos] = max_trans_score + emit_score #! Atention 
                                                                             #~ max trans score is also including the previos label score
                back_path_mat[cur_label_idx][pos] = max_trans_score_cor_label_idx
        #! find the label of the end pos which has the max score
        max_score = NEG_INF
        max_score_cor_label_idx = None
        for l_idx in range(label_num) :
            score = score_mat[l_idx][instance_len-1]
            if score > max_score :
                max_score = score
                max_score_cor_label_idx = l_idx
        #! build the revered predict tag list and coresponding instance vector list
        predict_tags_list_r = [model.trans_idx2label(max_score_cor_label_idx)]
        for i in range(instance_len -1 , 0 , -1) : #!! ( len - 1 ) -> 1
            max_score_cor_label_idx = back_path_mat[max_score_cor_label_idx][i]
            #print "pos %d , char %s , label %s" %(i , instance[i] , max_score_cor_label_idx)
            predict_tags_list_r.append(model.trans_idx2label(max_score_cor_label_idx))
        predict_tags_list_r.reverse() #! reverse , in replace !

        return predict_tags_list_r #! in fact , it is not reverse any more
