import os
import sys
import loader
import random
import numpy as np

'''
>>> dataset manager
'''
class data_manager(object):

    '''
    >>> Constructor
    >>> hyper_params
        >>> src_folders: source folders/files
        >>> dest_folders: destination folders/files
        >>> dict_file: the dictionary file
    '''
    def __init__(self,hyper_params):
        self.src_folders=hyper_params['src_folders']
        self.dest_folders=hyper_params['dest_folders']
        assert(len(self.src_folders)==len(self.dest_folders))
        self.dict_file=hyper_params['dict_file']                        # dictionary file
        self.entity_name_file=hyper_params['entity_name_file']          # entity name list
        self.word_frequency_threshold=hyper_params['word_frequency_threshold'] if hyper_params.has_key('word_frequency_threshold') else 0       # least word frequency
        self.word_list_length=hyper_params['word_list_length'] if hyper_params.has_key('word_list_length') else 10000                           # upper bound of vocabulary size
        self.document_length_threshold=hyper_params['document_length_threshold'] if hyper_params.has_key('document_length_threshold') else 100  # longest document length
        self.sentence_length_threshold=hyper_params['sentence_length_threshold'] if hyper_params.has_key('sentence_length_threshold') else 100  # longest sentence length
        self.valid_word_num=0                                           # number of words whose frequency is higher than threshold
        self.valid_word_list=[]                                         # valid word list whose length is min(self.word_list_length, self.valid_word_num)

        self.word_frequency=[]                      # list of [[word1, frequency1],[word2, frequency2] ...]
        self.entity_name_list=[]
        self.max_length_sentence=0
        self.max_length_document=0
        self.src_file_list=[]                       # source files with extension '.summary'
        self.dest_file_list=[]                      # destination files with extension '.info'

        self.file_set={}                            # sets of files, like training and test set
        self.file_set_pt={}                         # points for each set of files

        # scanning the folder, build src/dest files dictionary, dest files are not created in this part
        for src_folder_or_file, dest_folder_or_file in zip(self.src_folders,self.dest_folders):
            if os.path.isdir(src_folder_or_file):
                for file in os.listdir(src_folder_or_file):
                    if os.path.isfile(src_folder_or_file+os.sep+file) and file.split('.')[-1] in ['summary',]:
                        self.src_file_list.append(src_folder_or_file+os.sep+file)
                        output_file='.'.join(file.split('.')[:-1])+'.info'
                        self.dest_file_list.append(dest_folder_or_file+os.sep+output_file)
            elif os.path.isfile(src_folder_or_file):
                if src_folder_or_file.split('.')[-1] in ['summary']:
                    self.src_file_list.append(src_folder_or_file)
                    output_file='.'.join(src_folder_or_file.split('.')[:-1])+'.info'
                    self.dest_file_list.append(output_file)
            else:
                print 'invalid file or directory %s'%src_folder_or_file

        print 'There are %d files detected'%len(self.src_file_list)

    def analyze_documents(self):
        '''
        >>> analyze the document list and build the word_frequency list
        '''
        for idx,file in enumerate(self.src_file_list):
            print 'Analyze the document %d/%d - %.1f%%\r'%(
                idx+1,len(self.src_file_list),float(idx+1)/float(len(self.src_file_list))*100),
            document=loader.parse_document(file)
            if len(document['sentences'])>self.max_length_document:
                self.max_length_document=len(document['sentences'])
            for sentence in document['sentences']:
                words=sentence.split(' ')
                if len(words)>self.max_length_sentence:
                    self.max_length_sentence=len(words)
                for word in words:
                    index=self.find_word(word)
                    if index==-1:
                        self.word_frequency.append([word,1])
                    else:
                        self.word_frequency[index][1]+=1
            for entity in document['entity2name']:
                self.entity_name_list.append(document['entity2name'][entity])

        self.entity_name_list=list(set(self.entity_name_list))
        print 'There are %d entity name in total'%(len(self.entity_name_list))

        self.word_frequency=sorted(self.word_frequency,lambda x,y: -1 if x[1]>y[1] else 1)
        while self.valid_word_num<len(self.word_frequency) and self.word_frequency[self.valid_word_num][1]>=self.word_frequency_threshold:
            if self.valid_word_num<self.word_list_length:
                self.valid_word_list.append(self.word_frequency[self.valid_word_num][1])
            self.valid_word_num+=1

        print 'The vocabulary size in the whole corpura is %d,'%len(self.word_frequency)
        print 'There are %d words whose frequency is above %d'%(self.valid_word_num,self.word_frequency_threshold)

    '''
    >>> load states from dictionary file 
    i.e restore the value of self.word_frequency, self.valid_word_num and self.entity_name_list
    from file self.dict_file, self.entity_name_file
    '''
    def load_dict(self):
        if not os.path.exists(self.dict_file):
            print 'Failed to load dictionary from %s: file not exists'%self.dict_file
            return False
        self.word_frequency=[]
        with open(self.dict_file,'r') as fopen:
            for idx,line in enumerate(fopen):
                if idx==0:
                    self.max_length_document,self.max_length_sentence=map(int,line.split(' '))
                else:
                    parts=line.split(' ')
                    frequency=int(parts[-1])
                    word=' '.join(parts[1:-1])
                    self.word_frequency.append([word,frequency])

        while self.valid_word_num<len(self.word_frequency) and self.word_frequency[self.valid_word_num][1]>=self.word_frequency_threshold:
            if self.valid_word_num<self.word_list_length:
                self.valid_word_list.append(self.word_frequency[self.valid_word_num][1])
            self.valid_word_num+=1
        print 'Load %d words from %s'%(len(self.word_frequency),self.dict_file)
        print 'There are %d words whose frequency is above %d'%(self.valid_word_num,self.word_frequency_threshold)

        self.entity_name_list=open(self.entity_name_file,'r').readlines()
        self.entity_name_list=map(lambda x: x[:-1] if x[-1]=='\n' else x, self.entity_name_list)
        return True

    '''
    >>> build word index file for document
    >>> build dictionary file for int-str mapping
    >>> force: boolean, whether or not to overwrite existing files
    >>> output:
        >>> dict_file with each line 'INDEX:WORD:FREQUENCY'
        >>> entity_name_file with each line 'ENTITY_NAME'
        >>> input matrix and masks:
            first line: number of sentences NUMBER
            next NUMBER lines: NUMBER sentences
            last line: labels
    '''
    def build_files(self,force=False):
        # generate dictionary file
        print 'Generate dictionary file'
        if os.path.exists(self.dict_file) and force==False:
            print 'Dictionary file %s already exists. To overwrite it, please set force flag to True'%self.dict_file
        else:
            if not os.path.exists(os.path.dirname(self.dict_file)):
                os.makedirs(os.path.dirname(self.dict_file))
            with open(self.dict_file,'w') as fopen:
                fopen.write('%d %d\n'%(self.max_length_document,self.max_length_sentence))
                for idx,(word,frequency) in enumerate(self.word_frequency):
                    if (idx+1)%1000==0:
                        print '%d/%d ...\r'%(idx+1,len(self.word_frequency)),
                    fopen.write('%d %s %d\n'%(idx,word,frequency))
        print 'Completed!!         '

        print 'Generate entity dictionary file'
        if os.path.exists(self.entity_name_file) and force==False:
            print 'Entity name file %s already exists. To overwrite it, please set force flag to True'%self.entity_name_file
        else:
            if not os.path.exists(os.path.dirname(self.entity_name_file)):
                os.makedirs(os.path.dirname(self.entity_name_file))
            with open(self.entity_name_file,'w') as fopen:
                for entity in self.entity_name_list:
                    fopen.write('%s\n'%entity)
        print 'Completed!!         '

        # generate input matrix and labels
        print 'Generate input matrix and labels'
        for idx,(src_file,dest_file) in enumerate(zip(self.src_file_list, self.dest_file_list)):
            print '%d/%d ...\r'%(idx+1,len(self.src_file_list)),
            if os.path.exists(dest_file) and force==False:
                print 'Information file %s already exists. To overwrite it, please set force flag to True'%dest_file
            else:
                if not os.path.exists(os.path.dirname(dest_file)):
                    os.makedirs(os.path.dirname(dest_file))
                with open(dest_file,'w') as fopen:
                    document=loader.parse_document(src_file)
                    fopen.write(str(len(document['sentences']))+'\n')
                    # write word index for each sentence
                    for sentence in document['sentences']:
                        words=sentence.split(' ')
                        word_idx_list=map(lambda x: str(self.find_word(x)), words)
                        fopen.write(','.join(word_idx_list)+'\n')
                    fopen.write(','.join(map(str,document['labels'])))
        print 'Completed!!         '

    '''
    >>> initialization of data batch for a set of files
    >>> set_label: str, name of the file set
    >>> file_list: list<str> or None, name of files
    >>> permutation: bool, whether or not to permute the documents
    '''
    def init_batch_gen(self,set_label,file_list,permutation):
        if self.file_set.has_key(set_label) and file_list!=None:
            raise Exception('Can not initialize the dataset %s twice'%set_label)

        if file_list!=None:
            self.file_set[set_label]=[]
            for file_or_folder in file_list:
                if os.path.isdir(file_or_folder):
                    for file in os.listdir(file_or_folder):
                        if file.split('.')[-1] in ['info',]:
                            self.file_set[set_label].append(file_or_folder+os.sep+file)
                elif os.path.isfile(file_or_folder):
                    if file_or_folder.split('.')[-1] in ['info',]:
                        self.file_set[set_label].append(file_or_folder)

        self.file_set[set_label]=np.array(self.file_set[set_label])
        self.file_set_pt[set_label]=0
        if permutation==True:
            self.file_set[set_label]=np.random.permutation(self.file_set[set_label])

    '''
    >>> get the training data, including input matrix, masks and labels
    >>> set_label: str, set label e.g. 'train','validate'
    >>> batch_size: int, batch size
    >>> label policy: str in ['min','max','clear'], control if ambiguous sentences are not extracted/extracted/dropped
    >>> outputs:
        >>> input matrix: [batch_size, max_document_length, max_sentence_length]
        >>> masks: [batch_size, max_document_length]
        >>> labels: [batch_size, max_document_length]
    '''
    def batch_gen(self,set_label,batch_size,label_policy):
        if not label_policy in ['min','max','clear']:
            raise ValueError('Unrecognized labeling policy %s'%label_policy)
        if not self.file_set.has_key(set_label):
            raise ValueError('Set %s has not been initialized yet'%set_label)
        if batch_size>len(self.file_set[set_label]):
            raise ValueError('Too large batch size %d, there are %d documents in total'%(batch_size,len(self.file_set[set_label])))

        new_epoch=False
        if self.file_set_pt[set_label]+batch_size>len(self.file_set[set_label]): # Reach the end of the corpus
            self.init_batch_gen(set_label=set_label,file_list=None,permutation=True)
            new_epoch=True
        input_matrix=np.zeros([batch_size,self.document_length_threshold,self.sentence_length_threshold],dtype=np.int)
        input_matrix.fill(self.word_list_length+1)                                 # Padding is set to self.
        masks=np.zeros([batch_size,self.document_length_threshold],dtype=np.int)
        labels=np.zeros([batch_size,self.document_length_threshold],dtype=np.int)

        for batch_idx in xrange(batch_size):
            dest_file=self.dest_file_list[batch_idx+self.file_set_pt[set_label]]
            lines=open(dest_file,'r').readlines()
            lines=map(lambda x: x[:-1] if x[-1]=='\n' else x, lines)
            labels_this_document=map(int,lines[-1].split(','))
            number_of_sentences=int(lines[0])
            assert(number_of_sentences+2==len(lines))

            offset=0                                    # Cut short documents which is too long
            if number_of_sentences>self.document_length_threshold:
                offset=random.randint(0,number_of_sentences-self.document_length_threshold)
                labels_this_document=labels_this_document[offset:offset+self.document_length_threshold]
                number_of_sentences=self.document_length_threshold

            masks_this_document=np.ones([number_of_sentences],dtype=int)
            for sentence_idx in xrange(number_of_sentences):
                sentence=lines[sentence_idx+1+offset]
                word_idx_list=map(int,sentence.split(','))
                word_idx_list=map(lambda x:x if x<min(self.valid_word_num,self.word_list_length) else self.word_list_length, word_idx_list)
                if len(word_idx_list)>self.sentence_length_threshold:
                    sentence_offset=random.randint(0,len(word_idx_list)-self.sentence_length_threshold)
                    word_idx_list=word_idx_list[sentence_offset:sentence_offset+self.sentence_length_threshold]

                input_matrix[batch_idx,sentence_idx,:len(word_idx_list)]=word_idx_list
            for idx,(label,mask) in enumerate(zip(labels_this_document,masks_this_document)):
                if label==2:
                    if label_policy in ['min',]:
                        labels_this_document[idx]=0
                    elif label_policy in ['max',]:
                        labels_this_document[idx]=1
                    elif label_policy in ['clear',]:
                        masks_this_document[idx]=0

            masks[batch_idx,:number_of_sentences]=masks_this_document               # create the mask for this sentence
            labels[batch_idx,:number_of_sentences]=labels_this_document

        self.file_set_pt[set_label]+=batch_size
        return input_matrix,masks,labels,new_epoch

    '''
    >>> find a word in the dictionary, if not exist, return -1
    '''
    def find_word(self,word2find):
        for idx,(word,frequency) in enumerate(self.word_frequency):
            if word==word2find:
                return idx
        return -1
