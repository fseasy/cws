#coding=utf-8

import logging
from collections import Counter

from datasethandler import DatasetHandler
from symbols import INF , NEG_INF , TAG_B , TAG_M , TAG_E , TAG_S , TAG_NAME_TRANS 
from wsatom_char import WSAtomTranslator
from extractor import Extractor
from model import Model

logger = logging.getLogger("segmentor")
DEBUG=True

class Segmentor(object) :

    def __init__(self) :
        self.raw_training_data = None
        self.inner_lexicon = None 
        self.training_unigrams_data = None
        self.training_tags_data = None
        self.extractor = None
        self.model = None

    def train(self,training_f) :
        self.raw_training_data = DatasetHandler.read_training_data(training_f)
        self._build_inner_lexicon(threshold=0.9)
        self._processing_raw_training_data2unigrams_and_tags()
        self.extractor = Extractor(self.inner_lexicon)
        self._build_train_feature_space()
    
    def _build_train_feature_space(self) :
        if self.extractor is None or self.model is None  :
            return


    def _build_inner_lexicon(self , threshold=1.) :
        if self.raw_training_data is None :
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
        if DEBUG :
            logger.debug("origin words count : " + str(len(words_counter)))
            logger.debug("lexicon count : " + str(len(lexicon_list)))
        return dict.fromkeys(lexicon_list) #! to make it more efficient 
    
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
        if self.raw_training_data is None : return
        self.training_unigrams_data = []
        self.training_tags_data = []
        for sentence in self.raw_training_data :
            tags = []
            unigram_line = []
            for atom_ngram in sentence :
                partial_tags = Segmentor.__innerfunc__word2tags(atom_ngram)
                tags.extend(partial_tags)
                unigram_line.extend(atom_ngram) # atom_ngram is the list of WSAtom. so WSAtom(unigram) is added .
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

