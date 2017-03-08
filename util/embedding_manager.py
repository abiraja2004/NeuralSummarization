import os
import numpy as np
import data_manager

from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import euclidean_distances

'''
>>> word embedding manager
'''
class embedding_manager(object):

    '''
    >>> Constructor
    >>> hyper_params
        >>> embedding_dim: int, dimension of word embeddings
    '''
    def __init__(self,hyper_params):
        self.embedding_dim=hyper_params['embedding_dim']
        self.embedding_dict={}
        self.source=None

    '''
    >>> load embeddings
    >>> source: str, file to be loaded
    >>> format: str in ['text','bin'], the format of word embeddings in the file
    >>> force: boolean, whether or not to overwrite the existing word embeddings
    '''
    def load_embedding(self,source,format,force=False):
        if self.source!=None or self.embedding_dict!={} and force==False:
            raise Exception('This embedding manager has already read some embeddings. To overwrite it, please set force flag to True')

        if format=='text':
            lines=open(source,'r').readlines()
            lines=map(lambda x: x[:-1] if x[-1]!='\n' else x,lines)
            for idx,line in enumerate(lines):
                print 'Loading embeddings %d/%d\r'%(idx+1,len(lines)),
                parts=line.split(' ')
                word=parts[0]
                embeddings=np.array(map(float,parts[1:]))
                try:
                    assert(len(embeddings)==self.embedding_dim)
                except:
                    print 'line %d: dimension does not match, %d required but %d detected'%(idx+1,self.embedding_dim,len(embeddings))
                if self.embedding_dict.has_key(word):
                    print 'warning: word %s occurs more than one time in file %s, only keep the latest one'%(word,source)
                self.embedding_dict[word]=embeddings
        elif format=='bin':
            with open(source,'r') as fopen:
                header=fopen.readline()
                vocabulary_size,dimension=map(int,header.split())
                binary_length=np.dtype('float32').itemsize*dimension
                assert(dimension==self.embedding_dim)
                for word_idx in xrange(vocabulary_size):
                    print 'Loading embeddings %d/%d\r'%(word_idx+1,vocabulary_size),
                    word=[]
                    while True:
                        ch=fopen.read(1)
                        if ch==' ':
                            word=''.join(word)
                            break
                        else:
                            word.append(ch)
                    if self.embedding_dict.has_key(word):
                        print 'warning: word %s occurs more than one time in file %s, only keep the lastest one'%(word,source)
                    self.embedding_dict[word]=np.fromstring(fopen.read(binary_length),dtype=np.float32)
        else:
            raise ValueError('Unrecognized word embedding source format: %s'%format)
        self.source=source
        print 'Completed!!          '

    '''
    >>> generate embedding matrix for a dataset
    >>> manager: data_manager.data_manager, data manager which contains a word list
    '''
    def gen_embedding_matrix(self,manager):
        embedding_matrix=np.zeros([len(manager.word_frequency),self.embedding_dim],dtype=np.float32)
        missing_word_num=0
        print 'Generating embedding matrix'
        for idx,(word,frequency) in enumerate(self.word_frequency):
            print '%d/%d ... %d word Unrecognized\r'%(idx+1,len(self.word_frequency),missing_word_num),
            if self.embedding_dict.has_key(word):
                embedding_matrix[idx]=self.embedding_dict[word]
            else:
                embedding_matrix[idx]=np.random.randn([self.embedding_dim,])*0.5
                missing_word_num+=1
        return embedding_matrix

    '''
    >>> project the high dimensional embeddings
    >>> dim: int, dimension
    >>> method: str in ['pca','mds'], projection method
    '''
    def project(self,dim,method):
        if not method.lower() in ['pca','mds']:
            raise ValueError('Unrecognized projection method %s'%method)

        print 'Project word embeddings from %d dimension to %d dimension via %s'%(self.embedding_dim,dim,method)
        word_list=self.embedding_dict.keys()
        embedding_matrix=np.zeros([len(word_list),self.embedding_dim],dtype=np.float32)
        for idx,word in enumerate(word_list):
            embedding_matrix[idx]=self.embedding_dict[word]

        if method.lower() in ['pca']:
            pca=PCA(n_components=dim)
            embedding_matrix=pca.fit_transform(embedding_matrix)
        elif method.lower() in ['mds']
            distance_matrix=euclidean_distances(embedding_matrix)
            distance_matrix=(distance_matrix+distance_matrix.T)/2.0         # To make the distance matrix symmetric
            seed=np.random.RandomState(seed=3)
            mds=manifold.MDS(n_components=dim,max_iter=3000,eps=1e-9,random_state=seed,dissimilarity='precomputed',n_jobs=1)
            embedding_matrix=mds.fit(distance_matrix).embedding_

        for idx,word in enumerate(word_list):
            self.embedding_dict[word]=embedding_matrix[idx]
        self.embedding_dim=dim






