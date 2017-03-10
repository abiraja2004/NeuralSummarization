import os
import sys
import numpy as np

sys.path.insert(0,'util')
sys.path.insert(0,'model')

import network
import data_manager
import embedding_manager
import loader
import xml_parser

if len(sys.argv)!=2:
    print 'Usage: python sentence_extract.py <config>'
    exit(0)

hyper_params=xml_parser.parse(file=sys.argv[1],flat=False)

# Process dataset
data_process_params=hyper_params['data_process']
force_flag=data_process_params['force']
data_manager_params=data_process_params['data_manager_params']

my_data_manager=data_manager.data_manager(data_manager_params)
my_data_manager.load_dict()
# my_data_manager.scan_folders()
# my_data_manager.build_files(force=force_flag)

# Process word embedding
embedding_params=hyper_params['embedding']
embedding_manager_params=embedding_params['embedding_manager']
source=embedding_params['source']
format=embedding_params['format']
force=embedding_params['force']

my_embedding_manager=embedding_manager.embedding_manager(embedding_manager_params)
my_embedding_manager.load_embedding(source=source,format=format,force=force)
embedding_matrix=my_embedding_manager.gen_embedding_matrix(my_data_manager)

# Constructing the neural network
network_params=hyper_params['network']
sentence_extract_model_params=network_params['sentence_extract_model']

sentence_extract_model_params['sequence_length']=my_data_manager.max_length_sentence
sentence_extract_model_params['sequence_num']=my_data_manager.max_length_document
sentence_extract_model_params['vocab_size']=len(my_data_manager.word_frequency)
sentence_extract_model_params['embedding_dim']=my_embedding_manager.embedding_dim

my_network=network.sentenceExtractorModel(sentence_extract_model_params)

for batch_idx in xrange(400):
    input_matrix,masks,labels=my_data_manager.batch_gen(my_network.batch_size,label_policy='min')

    my_network.train_validate_test_init()
    prediction_this_batch, loss=my_network.train(input_matrix,masks,labels)
    print prediction_this_batch, loss
