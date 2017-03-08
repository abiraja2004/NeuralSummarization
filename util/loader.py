
'''
>>> parse documens in dailymail corpus
>>> return the following information: url, document sentences, the labels for these sentences, hightlights, entity2name mapping
'''
def parse_document(file_name):
    contents=open(file_name,'r').readlines()
    contents=''.joins(contents)
    parts=contents.split('\n\n')
    if len(parts)<4:
        print 'invalid file format in file: %s'%file_name
        return
    elif len(parts)>4:
        print 'weired file format in file: %s'%file_name
        print 'this file has %d parts'%len(parts)
    url,sentence_label,highlights,entity_map=parts[:4]

    url=map(lambda x: x[:-1] if x[-1]=='\n' else x, url.split('\n'))
    
    sentence_label=map(lambda x: x[:-1] if x[-1]=='\n' else x, sentence_label.split('\n'))
    sentences=map(lambda x: x.split('\t')[0], sentence_label)
    labels=map(lambda x: int(x.split('\t')[-1]), sentence_label)

    highlights=map(lambda x: x[:-1] if x[-1]=='\n' else x, highlights.split('\n'))

    entity_map=map(lambda x: x[:-1].split(':') if x[-1]=='\n' else x.split(':'), entity_map.split('\n'))
    entity2name={}
    for entity,name in entity_map:
        entity2name[entity]=name

    return {'url':url, 'sentences':sentences, 'labels':labels, 'highlights':highlights, 'entity2name':entity2name}

# '''
# >>> building properties for a corpus
# >>> return the maximum length of sentences, vocabulary set
# '''
# def parse_corpus(document_list):
#     word_list=[]
#     max_length=0
#     for idx,document in enumerate(document_list):
#         for sentence in document['sentences']:
#             words=sentence.split(' ')
#             if len(words)>max_length:
#                 max_length=len(words)
#             for word in words:
#                 if word_list.count(word)==0:
#                     word_list.append(word)

#     return max_length, word_list
