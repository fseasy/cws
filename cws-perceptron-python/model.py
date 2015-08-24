#coding=utf-8

TAG_LIST_NAME = [ "B" , "M" , "E" , "S" ]
TAG_B = 0
TAG_M = 1
TAG_E = 2 
TAG_S = 3

def word2tags(word) :
    wordlen = len(word)
    tags = [] 
    if wordlen = 1 :
        tags.append(TAG_S)
    elif wordlen >= 2 :
        tags.append(TAG_B)
        for i in range(1 , wordlen - 1) :
            tags.append(TAG_M)
        tags.append(TAG_E)  
    return tags 

def ready_training_data_from_lines(lines) :
    '''
    from lines(read from training data file )to trainning data(ws needed)
    Args :
        lines : list , each one is a str consisting of many words , unicode encoding

    Returns :
        zip(lines , tags_list)

    '''
    tags_list = []
    for line in lines :
        tags = []
        words = line.split()
        for word in words :
            partial_tags = word2tags(word)
            tags.extend(partial_tags)
        tags_list.append(tags)
    return zip(lines , tags_list)