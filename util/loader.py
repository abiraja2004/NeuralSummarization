import traceback

'''
>>> parse documens in dailymail corpus
>>> return the following information: url, document sentences, the labels for these sentences, hightlights, entity2name mapping
'''
def parse_document(file_name):
    try:
        contents=open(file_name,'r').readlines()
        contents=''.join(contents)
        parts=contents.split('\n\n')
        if len(parts)<4:
            print 'invalid file format in file: %s'%file_name
            return
        elif len(parts)>4:
            print 'weired file format in file: %s'%file_name
            print 'this file has %d parts'%len(parts)
        url,sentence_label,highlights,entity_map=parts[:4]

        url=map(lambda x: x[:-1] if x[-1]=='\n' else x, url.split('\n'))

        def split_entity_name(line):
            segments=line.split(':') if line[-1]!='\n' else line[:-1].split(':')
            return [segments[0],':'.join(segments[1:])]

        entity_map=map(split_entity_name, entity_map.split('\n'))
        entity2name={}
        entity_map=sorted(entity_map,lambda x,y: 1 if len(x)<len(y) else -1)
        for entity,name in entity_map:
            entity2name[entity]=name
            sentence_label.replace(entity,name)

        sentence_label=map(lambda x: x[:-1] if x[-1]=='\n' else x, sentence_label.split('\n'))
        sentences=map(lambda x: x.split('\t')[0], sentence_label)
        labels=map(lambda x: int(x.split('\t')[-1]), sentence_label)

        highlights=map(lambda x: x[:-1] if x[-1]=='\n' else x, highlights.split('\n'))

        return {'url':url, 'sentences':sentences, 'labels':labels, 'highlights':highlights, 'entity2name':entity2name}
    except:
        traceback.print_exc()
        raise Exception('Error occurs when parsing file: %s'%file_name)
