#coding=utf-8

import logging
from collections import Counter

from datasethandler import DatasetHandler
from symbols import INF , NEG_INF , TAG_B , TAG_M , TAG_E , TAG_S , TAG_NAME_TRANS , DEBUG
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

    def train(self,training_f,model_saving_f,max_iter=None) :
        self._set_max_iter(max_iter)
        self.raw_training_data = DatasetHandler.read_training_data(training_f)
        self._build_inner_lexicon(threshold=0.9)
        self._processing_raw_training_data2unigrams_and_tags()
        self._build_extractor()
        self._build_constrain()
        self._build_decoder()
        self._build_training_model()
        self._training_processing(model_saving_f)

    def _training_processing(self,model_saving_f) :
        '''
        Training
        '''
        logging.info("do training processing .")
        if self.training_unigrams_data is None or self.model is None or self.extractor is None :
            logging.error("failed!")
            return
        instance_num = len(self.training_unigrams_data)
        processing_print_interval = instance_num / 10 
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
            logging.info("Ite %d : %d instance processed. (%d%% / %d%%)" %( ite + 1 , instance_num ,
                          100 , float(ite+1) / self.max_iter * 100 ))
        logging.info("Training done.")
        self.model.save(model_saving_f)

    def _evaluate_processing(self , dev_path) :
        try :
            df = open(dev_path)
        except IOError , e :
            logging.error("Failed to load developing data from '%s'" %(dev_path))
            logging.info("Exit")
            exit(1)
        for instance in DatasetHandler.read_dev_data(df) :
            unigrams , tags = Segmentor._processing_one_segmented_WSAtom_instance2unigrams_and_tags(instance)

        df.close()

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
                           "minimun frequency in lexicon is %d , frequency ratio is %.2f%% , word count ratio is %.2f%%" %( 
                            threshold_num , freq_in_lexicon , total_freq , min_freq , 
                            freq_in_lexicon / float(total_freq) , len(lexicon_list) / float(len(words_counter)) )) 
                        )
        self.inner_lexicon =  dict.fromkeys(lexicon_list) #! to make it more efficient 
    
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
                         [ atom.get_combined_unicode_list() for atom in self.training_unigrams_data[0]] ).encode('utf8') ))
            logger.debug("the 1st tag list : " + " ".join([ TAG_NAME_TRANS[tag] for tag in self.training_tags_data[0] ]))
            logger.debug("the 1st origin seg line : " + " ".join(
                         [WSAtomTranslator.trans_atom_gram_list2unicode_line(atom_list).encode("utf8") 
                         for atom_list in self.raw_training_data[0]]))
    
    @staticmethod
    def _processing_one_segmented_WSAtom_instance2unigrams_and_tags(instance) :
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

