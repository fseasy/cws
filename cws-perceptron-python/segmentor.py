#coding=utf-8
try :
    import cPickle as pickle
except :
    import pickle
import os
import sys
import logging
import gzip
from collections import Counter

from datasethandler import DatasetHandler
from symbols import INF , NEG_INF , TAG_B , TAG_M , TAG_E , TAG_S , TAG_NAME_TRANS , DEBUG , OUTPUT_SEPARATOR
from wsatom_char import WSAtomTranslator
from extractor import Extractor
from model import Model
from constrain import Constrain
from decoder import Decoder

logger = logging.getLogger("segmentor")

class Segmentor(object) :

    def __init__(self) :
        self.raw_training_data = None
        self.inner_lexicon = None 
        self.training_unigrams_data = None
        self.training_tags_data = None
        
        self.max_iter = 5

        self.extractor = None
        self.constrain = None
        self.model = None
        self.decoder = None

    def train(self , training_path , dev_path , model_saving_path , max_iter=None) :
        self._set_max_iter(max_iter)
        self.raw_training_data = DatasetHandler.read_training_data(training_path)
        self._build_inner_lexicon(threshold=0.9)
        self._processing_raw_training_data2unigrams_and_tags()
        self._build_extractor()
        self._build_constrain()
        self._build_decoder()
        self._build_training_model()
        self._training_processing( model_saving_path , dev_path)

    def predict(self , model_path , predict_path , output_path ) :
        self._build_inner_lexicon_and_predict_model_from_saving_path(model_path)
        self._build_extractor()
        self._build_constrain() #! an empty constrain
        # self._build_decoder() #!! no need . We just use the Static Method of Decoder Class
        self._predict_processing(predict_path , output_path)
    
    def evaluate(self , gold_path , predict_path) :
        gold_ite = DatasetHandler.read_dev_data(gold_path)
        predict_ite = DatasetHandler.read_dev_data(predict_path)
        nr_processing = 0
        nr_gold = 0
        nr_processing_right = 0
        nr_line = 0
        while True :
            try :
                gold_instance = gold_ite.next()
                predict_instance = predict_ite.next()
            except StopIteration :
                break
            nr_line += 1
            gold_unigrams , gold_tags = self._processing_one_segmented_WSAtom_instance2unigrams_and_tags(gold_instance)
            predict_unigrams , predict_tags = self._processing_one_segmented_WSAtom_instance2unigrams_and_tags(predict_instance)
            gold_coor_seq = self.__innerfunc_4evaluate_generate_word_coordinate_sequence_from_tags(gold_tags)
            predict_coor_seq = self.__innerfunc_4evaluate_generate_word_coordinate_sequence_from_tags(predict_tags)
            cur_nr_gold , cur_nr_processing , cur_nr_processing_right = (
                    self.__innerfunc_4evaluate_get_nr_gold_and_processing_and_processing_right(gold_coor_seq , predict_coor_seq) )
            nr_gold += cur_nr_gold
            nr_processing += cur_nr_processing
            nr_processing_right += cur_nr_processing_right
        p , r , f = self.__innerfunc_4evaluate_calculate_prf(nr_gold , nr_processing , nr_processing_right)
        print ("Eval result :\np : %.2f%% r : %.2f%% f : %.2f%%\n"
              "line num : %d total word num : %d total predict word num : %d predict right num : %d ") %(
                p * 100 , r * 100, f * 100 , nr_line , nr_gold , nr_processing , nr_processing_right
                )

    def _training_processing(self , model_saving_path , dev_path) :
        '''
        Training
        '''
        logging.info("do training processing .")
        if self.training_unigrams_data is None or self.model is None or self.extractor is None :
            logging.error("failed!")
            return
        instance_num = len(self.training_unigrams_data)
        processing_print_interval = instance_num / 10 
        if processing_print_interval == 0 :
            processing_print_interval = 1 
        best_f = NEG_INF
        best_ite = -1
        best_model_data = None
        for ite in range(self.max_iter) :
            logging.info("training iteration %d ." %(ite + 1))
            for instance_id in range(instance_num) :
                instance = self.training_unigrams_data[instance_id]
                tags = self.training_tags_data[instance_id]
                predicted_tags = self.decoder.decode(self.extractor , self.model , self.constrain , instance )
                assert(len(tags) == len(predicted_tags))
                gold_features = self.decoder.calculate_label_sequence_feature(tags , instance , self.extractor , self.model)
                predicted_features = self.decoder.get_current_predict_label_sequence_feature()
                self.model.update_model(gold_features , predicted_features)
                #logging
                if ( instance_id + 1 ) % processing_print_interval == 0 :
                    current_ite_percent = ( instance_id + 1 ) / processing_print_interval * 10 
                    logging.info("Ite %d : %d instance processed. (%d%% / %d%%)" %( ite + 1 , instance_id + 1 ,
                                  current_ite_percent , current_ite_percent / self.max_iter +  float(ite) / self.max_iter * 100  ))
            logging.info("Ite %d done . %d instance processed. (%d%%)" %( ite + 1 , instance_num ,
                         float(ite+1) / self.max_iter * 100 ))
            f = self._4training_evaluate_processing(dev_path)
            #! save temporary model if best
            if f > best_f :
                best_f = f 
                best_ite = ite
                best_model_data = self.model.get_current_saving_data()
                logging.info("currently iteration %d get the best f1-score" %(ite + 1))
        logging.info("Training done.")
        logging.info("saving model at iteration %d with has best f1-score %.2f%%" %( best_ite + 1 , best_f * 100))
        self._save(model_saving_path , best_model_data )

    def _4training_evaluate_processing(self , dev_path) :
        nr_processing_right = 0
        nr_gold = 0
        nr_processing = 0
        for instance in DatasetHandler.read_dev_data(dev_path) :
            unigrams , gold_tags = Segmentor._processing_one_segmented_WSAtom_instance2unigrams_and_tags(instance)
            predict_tags = Decoder.decode_for_predict(self.extractor , self.model , self.constrain , unigrams)
            gold_coor_seq = self.__innerfunc_4evaluate_generate_word_coordinate_sequence_from_tags(gold_tags)
            predict_coor_seq = self.__innerfunc_4evaluate_generate_word_coordinate_sequence_from_tags(predict_tags)
            cur_nr_gold , cur_nr_processing , cur_nr_processing_right = (
                            self.__innerfunc_4evaluate_get_nr_gold_and_processing_and_processing_right(gold_coor_seq , predict_coor_seq)
                    )
            nr_gold += cur_nr_gold
            nr_processing += cur_nr_processing
            nr_processing_right += cur_nr_processing_right
        p , r , f = self.__innerfunc_4evaluate_calculate_prf(nr_gold , nr_processing , nr_processing_right)
        print >>sys.stderr , ("Eval result :\np : %.2f%% r : %.2f%% f : %.2f%%\n"
               "total word num : %d total predict word num : %d predict right num : %d ")%(
                p * 100 , r * 100, f * 100 , nr_gold , nr_processing , nr_processing_right
                )
        return f
    
    def __innerfunc_4evaluate_generate_word_coordinate_sequence_from_tags(self , tags) :
        '''
        generate coordinate sequence from tags 
        => B M E S S              (tags)
        => (0,2) , (3,3) , (4,4)  (generate coordinate sequence)
        => (中国人）（棒）（棒）  (generate word sequence directly)
        that means , every coordiante stands for a word in the origin word sequence .
        do this , it [may be more convenient than genenrate word directly] from tags
        '''
        coor_seq = []
        start_idx = 0
        end_idx = 0
        for i in range(len(tags)) :
            tag = tags[i]
            if tag == TAG_E or tag == TAG_S :
                end_idx = i
                coor = (start_idx , end_idx)
                coor_seq.append(coor)
                start_idx = end_idx + 1
        return coor_seq

    def __innerfunc_4evaluate_get_nr_gold_and_processing_and_processing_right(self , gold_coor_seq , predict_coor_seq) :
        '''
        get nr_gold , nr_processing , nr_processing_right
        Args :
            gold_coor_seq : list , [ (0,2) , (3 , 3) , (4 , 4) ]
            predict_coor_seq : list , same as gold_coor_seq
        Returns :
            nr_gold , nr_processing , nr_processing_right : int
        '''
        nr_gold = len(gold_coor_seq)
        nr_processing = len(predict_coor_seq)
        i = j = 0
        nr_processing_right = 0
        while i < nr_gold and j < nr_processing :
            gold_coor = gold_coor_seq[i]
            processing_coor = predict_coor_seq[j]
            #! first , align the word start
            if gold_coor[0] < processing_coor[0] : 
                #! gold is behind , ++i
                i = i + 1
                continue
            elif gold_coor[0] > processing_coor[0] :
                #! processing is behind , ++j
                j = j + 1
                continue
            else :
                #! aligned , is right ?
                if gold_coor == processing_coor :
                    nr_processing_right += 1
                #! alwats ahead
                i += 1
                j += 1
        return nr_gold , nr_processing , nr_processing_right

    def __innerfunc_4evaluate_calculate_prf(self , nr_gold , nr_predict , nr_predict_right) :
        '''
        calculate precesion , recall , f_1-measure
        '''
        p = float(nr_predict_right) / nr_predict
        r = float(nr_predict_right) / nr_gold
        #f = 2 * p * r / (p + r) 
        f = 2 * nr_predict_right / float(nr_gold + nr_predict)
        return (p , r , f)
    
    def _predict_processing(self , predict_path , output_path) :
        if isinstance(output_path , file) :
            output_f = output_path 
        else :
            if  output_path == "stdout" :
                output_f = sys.stdout
            else :
                output_f = open(output_path , "w")
        logging.info("set output %s " %(output_f.name))
        logging.info("reading instance from %s . predicting ." %(predict_path))
        for instance , separator_data in DatasetHandler.read_predict_data(predict_path) :
            self.constrain.set_constrain_data(separator_data)
            predict_tags = Decoder.decode_for_predict(self.extractor , self.model , self.constrain , instance)
            segmented_line = self._processing_unigrams_and_tags2segmented_line(instance,predict_tags)
            output_f.write("%s" %( "".join([segmented_line , os.linesep]) ) )
        if output_f is not sys.stdout :
            output_f.close()
        logging.info("predicting done.")

    def _set_max_iter(self , max_iter) :
        if max_iter is None or type(max_iter) is not int or max_iter < 1 :
            logging.warning("Max iteration number is not set or in valid state .")
            logging.info("set it to default value .")
        else :
            self.max_iter = max_iter
        logging.info("Max iteration is %d ." %(self.max_iter))

    def _build_extractor(self) :
        self.extractor = Extractor(self.inner_lexicon)
    
    def _build_constrain(self) :
        self.constrain = Constrain()

    def _build_decoder(self) :
        self.decoder = Decoder()
    
    def _build_inner_lexicon_and_predict_model_from_saving_path(self , model_path) :
        self.model = Model()
        self._load(model_path)
        
    def _build_training_model(self) :
        '''
        init a empty model , build model emit feature space , build model label space , build model weight
        '''
        #! Init empty model
        logging.info("Initialize an empty model")
        self.model = Model()
        self.model.init_empty_model()
        #! build emit feature space
        logging.info("extract all training instance and build model feature space .")
        if self.extractor is None or self.model is None or self.training_unigrams_data is None :
            logging.error("failed!")
            return
        for atom_line in self.training_unigrams_data :
            emit_feature_list = self.extractor.extract_emit_features(atom_line)
            self.model.add_emit_feature_list2feature_space(emit_feature_list)
        #! build label space
        logging.info("add labels to model label space .")
        self.model.add_labels2label_space( (TAG_B , TAG_M , TAG_E , TAG_S) )
        #! build feature tans mat and weight
        logging.info("Inlitialize feature transition and weight space .")
        self.model.build_up_model()
        

    def _build_inner_lexicon(self , threshold=1.) :
        logging.info("build inner lexicon from training data .")
        if self.raw_training_data is None :
            logging.error('failed')
            return
        words_counter = Counter()
        for raw_instance in self.raw_training_data :
            unicode_instance = [ WSAtomTranslator.trans_atom_gram_list2unicode_line(atom_instance_gram_list) 
                                 for atom_instance_gram_list in raw_instance ]
            words_counter.update(unicode_instance)
        total_freq = sum(words_counter.viewvalues())
        lexicon_list = []
        if threshold < 1. :
            ##! a fast and clearly implementation is using Counter.most_common(N) to return the threshold number words . 
            ##! but it clearly will cause some words  were added to lexicon dict while some ohter words with the same freq is cut off at tail . it is bad.
            ##! So do following logic to keep fair .
            threshold_num = int( total_freq * threshold )
            pre_freq = INF
            words_has_same_freq = []
            freq_counter = 0
            for word , freq in words_counter.most_common() :
                if freq != pre_freq :
                    lexicon_list.extend(words_has_same_freq)
                    words_has_same_freq = []
                    pre_freq = freq
                words_has_same_freq.append(word)
                freq_counter += freq
                if freq_counter > threshold_num :
                    break
        else :
            lexicon_list = words_counter.keys()
        logging.info( "inner lexicon info : %d/%d" %( len(lexicon_list) , len(words_counter) )  )
        
        if DEBUG :
            freq_in_lexicon = 0
            min_freq = INF
            for word in lexicon_list :
                word_freq = words_counter[word]
                freq_in_lexicon += word_freq
                if word_freq < min_freq :
                    min_freq = word_freq
            logger.debug("origin words count : " + str(len(words_counter)))
            logger.debug("lexicon count : " + str(len(lexicon_list)))
            logger.debug( ("thredhold num is %d , actually total freqency in lexicon is %d(total frequency of all words : %d ),"
                           "minimun frequency in lexicon is %s , frequency ratio is %.2f%% , word count ratio is %.2f%%" %( 
                            threshold_num , freq_in_lexicon , total_freq , min_freq , 
                            freq_in_lexicon / float(total_freq) , len(lexicon_list) / float(len(words_counter)) )) 
                        )
        self.inner_lexicon =  dict.fromkeys(lexicon_list) #! to make it more efficient 
   
    def _processing_unigrams_and_tags2segmented_line(self , unigrams , tags) :
        '''
        trans unigrams and tags to segmented word line
        =>unigrams = [我 , 是 , 中 , 国 , 人] , tags = [S , S , B , E , S] => ret = 我 是 中国 人
        Actually , unigrams is a list of WSAtom , function `str` can trans it to str which is encoded in origin encoding.
        
        Args :
            unigrams : list , WSAtom list
            tags :  list , TAGs
        Returns :
            line , str encoded . segmented .
        '''
        if len(unigrams) == 0 :
            return "" #! An empty line is included !
        line = []
        for unigram , tag in zip(unigrams , tags) :
            line.append(str(unigram))
            if tag == TAG_E or tag == TAG_S :
                line.append(OUTPUT_SEPARATOR)
        return "".join(line[:-1]) #! the last will be a redundant OUTPUT_SEPARATOR char ( we assert that the last tag must be E or S )

    def _processing_raw_training_data2unigrams_and_tags(self) :
        '''
        from lines data(WSAtom wrapped )to trainning data(ws needed)
        [ inner class function ]
        logic :
            we process self.raw_training_data . 
            and set self.training_unigrams_data , self.training_tags_data
            unigram_line_list : list , elements is also a list . the most inner element is the unigram . 
                                => [ [WSAtom(unigram) , WSAtom(unigram) , ...] , ...  ] 
            tags_list : list of list . most inner element is tag . => [ [tag_b , tag_m , ...] , ...]
    
        '''
        logging.info("processing raw training data to unigrams and tags .")
        if self.raw_training_data is None : 
            logging.error("failed!")
            return
        self.training_unigrams_data = []
        self.training_tags_data = []
        for sentence in self.raw_training_data :
            unigram_line , tags = Segmentor._processing_one_segmented_WSAtom_instance2unigrams_and_tags(sentence)
            self.training_tags_data.append(tags)
            self.training_unigrams_data.append(unigram_line)
        if DEBUG :
            logger.debug("the 1st line : %s" %( u" ".join(
                         [ unicode(atom) for atom in self.training_unigrams_data[0]] ).encode('utf8') ))
            logger.debug("the 1st tag list : " + " ".join([ TAG_NAME_TRANS[tag] for tag in self.training_tags_data[0] ]))
            logger.debug("the 1st origin seg line : " + " ".join(
                         [WSAtomTranslator.trans_atom_gram_list2unicode_line(atom_list).encode("utf8") 
                         for atom_list in self.raw_training_data[0]]))
    
    @staticmethod
    def _processing_one_segmented_WSAtom_instance2unigrams_and_tags(instance) :
        '''
        from [ [ WSAtom , WSAtom , ...] , [ WSAtom , ... ] , ... ] to [ WSAtom , WSAtom , ... ] and [ TAG_X , TAG_X , ...  ]
        Args :
            instance : list of list , [ [ WSAtom , WSAtom , ... ]  , [ WSAtom , ...  ] , ... ]
        Returns :
            unigrams : list , [WSAtom , WSAtom , WSAtom , ... ]
            tags     : list , [TAG_X , TA_X , TAG_X , ...]
        Why Static Method ?
            no specific meaning . just to using the staticmethod function
        '''
        tags = []
        unigram_line = []
        for atom_ngram in instance :
            partial_tags = Segmentor.__innerfunc__word2tags(atom_ngram)
            tags.extend(partial_tags)
            unigram_line.extend(atom_ngram) # atom_ngram is the list of WSAtom. so WSAtom(unigram) is added .
        return unigram_line , tags

    @staticmethod
    def __innerfunc__word2tags(word) :
        wordlen = len(word)
        tags = [] 
        if wordlen == 1 :
            tags.append(TAG_S)
        elif wordlen >= 2 :
            tags.append(TAG_B)
            for i in range(1 , wordlen - 1) :
                tags.append(TAG_M)
            tags.append(TAG_E)  
        return tags

    def _save(self , fpath , best_model_data=None ) :
        logging.info("saving model to '%s'." %(fpath))
        zfo = gzip.open(fpath , "wb")
        #! saving inner lexicon
        pickle.dump(self.inner_lexicon , zfo)
        #! saving model parameter
        if best_model_data is None :
            self.model.save(zfo)
        else :
            pickle.dump(best_model_data, zfo)
        zfo.close()
        logging.info("saving done.")

    def _load(self , fpath) :
        logging.info("loading model from '%s'" %(fpath))
        zfi = gzip.open(fpath , "rb")
        #! load inner lexicon
        self.inner_lexicon = pickle.load(zfi)
        #! loading model parameter
        self.model.load(zfi)
        zfi.close()
        logging.info("loading done .")
    
