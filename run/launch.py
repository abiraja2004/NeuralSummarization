## This script is to lauch a summary generation system

import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
if sys.version_info.major==2:
    input=raw_input
sys.path.insert(0,'./util')
from py2py3 import *
import numpy as np

sys.path.insert(0,'model')
sys.path.insert(0,'util')

import network
import fasttext

import text_manager
import xml_parser

class solver(object):

    def __init__(self,hyper_params):
        self.network_type=hyper_params['type']
        if self.network_type.lower() in ['sentence_extraction',]:
            print('Loading a sentence extraction model')
            network_params=hyper_params[self.network_type]
            model2load=hyper_params['model2load']

            # Construct the network
            self.model=network.sentenceExtractorModel(network_params)
            self.model.train_validate_test_init()
            self.model.load_params(model2load)

        elif self.network_type.lower() in ['fasttext','fast_text']:
            # For fast text
            print('TODO')
        else:
            raise ValueError('Unrecognized network type: %s'%self.network_type)

        word_list_file=hyper_params['word_list_file']
        self.word_list=[]
        with open(word_list_file,'r') as fopen:
            fopen.readline()
            for line in fopen:
                parts=line.split(' ')
                word=' '.join(parts[1:-1])
                self.word_list.append(word)

    def extract(self,file_name,n_top=5):
        if self.network_type.lower() in ['sentence_extraction',]:
            return self.model.do_summarization([file_name,],self.word_list,n_top=n_top)[0]
        elif self.network_type.lower() in ['fasttext','fast_text']:
            # TO DO
            print('TODO')
        else:
            raise ValueError('Unrecognized network type: %s'%self.network_type)



if __name__=='__main__':
    if len(sys.argv)!=2:
        print 'Usage: python launch.py <config>'
        exit(0)

    hyper_params=xml_parser.parse(sys.argv[1],flat=False)

    mode=hyper_params['mode']
    
    if mode=='cmd_demo':
        solver_param=hyper_params['solver_param']
        local_params=hyper_params[mode]
        n_top=5 if not 'n_top' in local_params else local_params['n_top']

        my_solver=solver(solver_param)
        print('command line demo started!!')
        while True:
            print('type in the file to analysize, type "exit" to exit or "bash" to launch bash shell')
            answer=input('>>> ')
            if answer.lower() in ['sh','bash']:
                os.system('bash')
            elif answer.lower() in ['exit']:
                break
            else:
                if not os.path.exists(answer):
                    print('file not exists: %s'%answer)
                else:
                    results=my_solver.extract(file_name=answer,n_top=n_top)
                    sentences=open(answer,'r').readlines()
                    sentences=map(lambda x:x if x[-1]!='\n' else x[:-1], sentences)
                    print('===========Original Text===============')
                    for idx,sentence in enumerate(sentences):
                        print('%d\t%s'%(idx+1,sentence.encode('utf8')))
                    print('===============Summary=================')
                    for idx,pts,sentence in results:
                        print('%d\t%.3f\t%s'%(idx+1,pts,sentence.encode('utf8')))
                    print('=================End===================')
    elif mode=='web_demo':
        print('TODO')
    else:
        raise ValueError('Unrecognized mode: %s'%mode)


