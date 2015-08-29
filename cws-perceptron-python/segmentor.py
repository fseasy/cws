#coding=utf-8

import logging
from collections import Counter
from datasethandler import DatasetHandler
from symbols import INF , NEG_INF

logger = logging.getLogger("segmentor")
DEBUG=True

class Segmentor(object) :

    def __init__(self) :
        self.raw_training_data = None
        self.inner_lexicon = None 
        self.training_unigrams_data = None
        self.training_tags_data = None


    def train(self,training_f) :
        self.raw_training_data = DatasetHandler.read_training_data(training_f)
        self.build_inner_lexicon(threshold=0.9)
        
    def build_inner_lexicon(self , threshold=1.) :
        if self.raw_training_data is None :
            return
        words_counter = Counter()
        for raw_instance in self.raw_training_data :
            words_counter.update(raw_instance)
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
    
    def processing_raw_training_data2unigrams_and_tags(self) :
        '''
        from lines(read from training data file )to trainning data(ws needed)
        Args :
            lines : list , each one is a str consisting of many words , unicode encoding
    
        Returns :
            unigram_line_list : list , elements is also a list . the most inner element is the unigram . => [ [unigram , unigram , ...] , ...  ] 
            tags_list : list of list . most inner element is tag . => [ [tag_b , tag_m , ...] , ...]
    
        '''
        def word2tags(word) :
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
        self.training_unigrams_data = []
        self.training_tags_data = []
        for words_list in self.raw_training_data :
            tags = []
            unigram_line = []
            for word in words_list :
                partial_tags = word2tags(word)
                tags.extend(partial_tags)
                unigram_line.extend(word) # str is also iterable as for each unigram .  so unigram is added .
            self.training_tags_data.append(tags)
            self.training_unigrams_data.append(unigram_line)

