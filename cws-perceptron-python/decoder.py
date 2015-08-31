#coding=utf-8

from symbols import NEG_INF 
class Decoder(object) :
    
    @staticmethod
    def decode(extractor , model , constrain ,instance) :
        '''
        Vertebi Decode
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
        back_path_mat = [ [ None ] * instance_len for n in range(label_num ]
        #! instance vector mat . cached it for update .
        instance_vector_mat = [ [ None ] * instance_len for n in range(label_num)]

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
                    w = model.get_current_weight_vector()
                    trans_score = Tools.sparse_vector_dot(trans_vector , w )
                    pre_label_idx = model.trans_label2idx(pre_label)
                    pre_label_score = score_mat[pre_label_idx][pos-1] if pos -1 >= 0 else 0
                    tmp_score = trans_score + pre_label_score
                    if tmp_score > max_trans_score :
                        max_trans_score = tmp_score
                        max_trans_score_cor_label = pre_label_idx
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
            score = score_mat[l_idx][instance_len]
            if score > max_score :
                max_score = score
                max_score_cor_label_idx = l_idx
        #! build the revered predict tag list and corresponding instance vector list
        predict_tags_list_r = [model.trans_idx2label(max_score_cor_label_idx)]
        predict_instance_vector_list_r = [ instance_vector_mat[max_score_cor_label_idx][instance_len-1] ]
        for i in range(instance_len -1 , 0 , -1) : #!! ( len - 1 ) -> 1
            max_score_cor_label_idx = back_path_mat[max_score_cor_label_idx][i]
            predict_tags_list_r.append(model.trans_idx2label(max_score_cor_label_idx))
            predict_instance_vector_list_r.append(instance_vector_mat[max_score_cor_label_idx][i-1]) #!! i-1 
        predict_tags_list_r.reverse() #! reverse , in replace !
        predict_instance_vector_list_r.reverse()
        return predict_tags_list_r , predict_instance_vector_list_r
