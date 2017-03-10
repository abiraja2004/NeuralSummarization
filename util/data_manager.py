import os
import sys
import loader
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
        self.dict_file=hyper_params['dict_file']

        self.word_frequency=[]
        self.max_length_sentence=0
        self.max_length_document=0
        self.src_file_list=[]                       # source files with extension '.summary'
        self.dest_file_list=[]                      # destination files with extension '.info'
        self.index=0                                # current position, used for generating batch data

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
            print 'Analyze the document %d/%d ...\r'%(idx+1,len(self.src_file_list)),
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

        self.word_frequency=sorted(self.word_frequency,lambda x,y: -1 if x[1]>y[1] else 1)
        print 'The vocabulary size in the whole corpura is %d'%len(self.word_frequency)

    '''
    >>> load states from dictionary file
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
        print 'Load %d words from %s'%(len(self.word_frequency),self.dict_file)
        return True

    '''
    >>> build word index file for document
    >>> build dictionary file for int-str mapping
    >>> force: boolean, whether or not to overwrite existing files
    >>> output:
        >>> dict_file with each line 'INDEX:WORD:FREQUENCY'
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
    >>> initialization of data batch generation
    >>> permutation: boolean, whether or not to permute the document
    '''
    def init_batch_gen(self,permutation):
        self.src_file_list=np.array(self.src_file_list)
        self.dest_file_list=np.array(self.dest_file_list)
        if permutation==True:
            number_files=len(self.src_file_list)
            orders=np.arange(number_files)
            orders=np.random.permutation(orders)
            self.src_file_list=self.src_file_list[orders]
            self.dest_file_list=self.dest_file_list[orders]

    '''
    >>> get the training data, including input matrix, masks and labels
    >>> batch_size: int, batch size
    >>> label policy: str in ['min','max','clear'], control if ambiguous sentences are not extracted/extracted/dropped
    >>> outputs:
        >>> input matrix: [batch_size, max_document_length, max_sentence_length]
        >>> masks: [batch_size, max_document_length]
        >>> labels: [batch_size, max_document_length]
    '''
    def batch_gen(self,batch_size,label_policy):
        if not label_policy in ['min','max','clear']:
            raise ValueError('Unrecognized labeling policy %s'%label_policy)
        if batch_size>len(self.src_file_list):
            raise ValueError('Too large batch size %d, there are %d documents in total'%(batch_size,len(self.src_file_list)))
        if self.index+batch_size>len(self.src_file_list):           # Reach the end of the corpus
            self.index=0
            self.init_batch_gen(permutation=True)
        input_matrix=np.zeros([batch_size,self.max_length_document,self.max_length_sentence],dtype=np.int)
        masks=np.zeros([batch_size,self.max_length_document],dtype=np.int)
        labels=np.zeros([batch_size,self.max_length_document],dtype=np.int)

        for batch_idx in xrange(batch_size):
            dest_file=self.dest_file_list[batch_idx+self.index]
            lines=open(dest_file,'r').readlines()
            lines=map(lambda x: x[:-1] if x[-1]=='\n' else x, lines)
            number_of_sentences=int(lines[0])
            assert(number_of_sentences+2==len(lines))

            for sentence_idx in xrange(number_of_sentences):
                sentence=lines[sentence_idx+1]
                word_idx_list=map(int,sentence.split(','))
                input_matrix[batch_idx,sentence_idx,:len(word_idx_list)]=word_idx_list
            labels_this_sentence=map(int,lines[-1].split(','))
            masks_this_sentence=np.ones([number_of_sentences],dtype=int)
            for idx,(label,mask) in enumerate(zip(labels_this_sentence,masks_this_sentence)):
                if label==2:
                    if label_policy in ['min',]:
                        labels_this_sentence[idx]=0
                    elif label_policy in ['max',]:
                        labels_this_sentence[idx]=1
                    elif label_policy in ['clear',]:
                        masks_this_sentence[idx]=0

            masks[batch_idx,:number_of_sentences]=masks_this_sentence               # create the mask for this sentence
            labels[batch_idx,:number_of_sentences]=labels_this_sentence

        self.index+=batch_size
        return input_matrix,masks,labels

    '''
    >>> find a word in the dictionary, if not exist, return -1
    '''
    def find_word(self,word2find):
        for idx,(word,frequency) in enumerate(self.word_frequency):
            if word==word2find:
                return idx
        return -1
