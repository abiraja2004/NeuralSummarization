import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
sys.path.insert(0,'./util')
from py2py3 import *
import numpy as np

sys.path.insert(0,'util')
sys.path.insert(0,'model')

import fasttext
import data_manager
import embedding_manager
import xml_parser

if len(sys.argv)!=2:
    print('Usage: python fast_text.py <config>')
    exit(0)

hyper_params=xml_parser.parse(file=sys.argv[1],flat=False)

# Process dataset
data_process_params=hyper_params['data_process']
data_manager_params=data_process_params['data_manager_params']
file_sets=data_process_params['file_sets']

my_data_manager=data_manager.data_manager(data_manager_params)
my_data_manager.load_dict()
for key in file_sets:
    file_set=file_sets[key]
    my_data_manager.init_batch_gen(set_label=key, file_list=file_set, permutation=True)

# Process word embedding
embedding_params=hyper_params['embedding']
embedding_manager_params=embedding_params['embedding_manager']
source=embedding_params['source']
format=embedding_params['format']
force=embedding_params['force']

my_embedding_manager=embedding_manager.embedding_manager(embedding_manager_params)
my_embedding_manager.load_embedding(source=source,format=format,force=force)
embedding_matrix=my_embedding_manager.gen_embedding_matrix(my_data_manager)

# Construct the network
network_params=hyper_params['network']

fasttext_model_params=network_params['fasttext_model']
model2load=network_params['model2load']
# Sepecify some key parameters
assert(fasttext_model_params['sequence_length']==my_data_manager.sentence_length_threshold)
assert(fasttext_model_params['vocab_size']==my_data_manager.word_list_length)
assert(fasttext_model_params['embedding_dim']==my_embedding_manager.embedding_dim+my_data_manager.extended_bits)
if 'pretrain_embedding' in network_params and network_params['pretrain_embedding']==True:
    sentence_extract_model_params['embedding_matrix']=embedding_matrix

my_network=fasttext.fastText(fasttext_model_params)

test_case_num=0
test_right_num=0
positive_num=0
negative_num=0
my_network.train_validate_test_init()
my_network.load_params(model2load)
while True:
    input_matrix,masks,labels,stop=my_data_manager.batch_gen(set_label='test',batch_size=my_network.batch_size,label_policy='min',model_tag='fasttext')
    if stop==True:
        break

    predictions=my_network.test(input_matrix,masks)
    predictions=np.array(predictions).reshape(-1)
    labels=np.array(labels).reshape(-1)

    hits=map(lambda x: 1 if x[0]==x[1] else 0, zip(predictions,labels))
    positive_bit=map(lambda x: 1 if x==1 else 0, labels)
    negative_bit=map(lambda x: 1 if x==0 else 0, labels)

    test_case_num+=len(labels)
    test_right_num+=np.sum(hits)
    positive_num+=np.sum(positive_bit)
    negative_num+=np.sum(negative_bit)

    sys.stdout.write('test_accuracy=%d/%d=%.1f%%, positive=%d(%.1f%%), negative=%d(%.1f%%)\r'%(test_right_num,test_case_num,float(test_right_num)/float(test_case_num)*100,
        positive_num,float(positive_num)/float(test_case_num)*100,negative_num,float(negative_num)/float(test_case_num)*100))
print('')

