import numpy as np
import tensorflow as tf

class sentenceExtractorModel(object):

    def __init__(self, hyper_params):
        '''
        >>> Construct a CNN-LSTM model
        >>> hyper_params: dict, a dictionary containing all hyper parameters
            >>> batch_size: int, batch size
            >>> sequence_length: int, maximum sentence length
            >>> sequence_num: int, maximum number of sentences in a document
            >>> rnn_size: int, the number of neurons in LSTM
            >>> mlp_neurons: list<int>, the number of neurons in each layer of mlp
            >>> class_num: int, number of categories
            >>> vocab_size: int, vocabulary size
            >>> embedding_dim: int, dimension of word embeddings
            >>> filter_sizes: list<int>, different kinds of filter size i.e window size
            >>> feature_map: int, number of feature maps for different filters
            >>> update_policy: dict, update policy
            >>> embedding_matrix: optional, numpy.array, initial embedding matrix of size [self.vocab_size, self.embedding_dim]
        '''
        self.batch_size=hyper_params['batch_size']
        self.sequence_length=hyper_params['sequence_length']
        self.sequence_num=hyper_params['sequence_num']
        self.rnn_size=hyper_params['rnn_size']
        self.mlp_neurons=hyper_params['mlp_neurons']
        self.class_num=hyper_params['class_num']
        self.vocab_size=hyper_params['vocab_size']
        self.embedding_dim=hyper_params['embedding_dim']
        self.filter_sizes=hyper_params['filter_sizes']
        self.feature_map=hyper_params['feature_map']
        self.update_policy=hyper_params['update_policy']

        self.sess=None

        self.embedding_matrix=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_dim],-1.0,1.0),dtype=tf.float32) \
            if not hyper_params.has_key('embedding_matrix') else tf.Variable(hyper_params['embedding_matrix'],dtype=tf.float32)

        self.inputs=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_num,self.sequence_length]) # None represents the number of sequences
        self.masks=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_num])                        # Masks - 1 if the sentence is valid else 0
        self.labels=tf.placeholder(tf.int32,shape=[self.batch_size, self.sequence_num])                      # Labels of each sentences

        self.embedding_output=tf.nn.embedding_lookup(self.embedding_matrix,self.inputs)                      # shape = [self.batch_size, sequence_num, self.sequence_length, self.embedding_dim]

        input_slices=tf.split(self.embedding_output,self.sequence_num,1)

        # Construct sentence_embedding for each sentences in the documents
        sentence_embeddings=[]
        with tf.variable_scope('CNN') as scope:
            for slice_idx in xrange(self.sequence_num):
                if slice_idx!=0:
                    scope.reuse_variables()
                parts=[]
                for filter_idx, filter_size in enumerate(self.filter_sizes):
                    convpool_output=self.convpool(input_data_3d=input_slices[slice_idx],
                        filter_width=filter_size, name='filter%d'%filter_idx)       # of size [self.batch_size, self.feature_map]
                    parts.append(convpool_output)
                current_embedding=parts[0]
                for part in parts[1:]:
                    current_embedding=tf.add(part,current_embedding)
                sentence_embeddings.append(current_embedding)

        # LSTM-based document encoding part
        with tf.variable_scope('LSTM') as scope:
            LSTM_cell=tf.contrib.rnn.BasicLSTMCell(self.rnn_size)        # Contruct a LSTM cell
            state_sequences=[]
            state=LSTM_cell.zero_state(self.batch_size,tf.float32)
            for sentence_idx in xrange(self.sequence_num):
                if sentence_idx!=0:
                    scope.reuse_variables()
                _, state=LSTM_cell(sentence_embeddings[sentence_idx],state)
                state_sequences.append(state[1])                    # record the hidden state in each step

        # Sentence Extraction part
        predictions=[]
        expand_matrix=tf.ones([1,self.feature_map],dtype=tf.float32)
        buffer_input=tf.zeros([self.batch_size,self.feature_map],dtype=tf.float32)
        with tf.variable_scope('LSTM',reuse=True) as scope:
            _, state=LSTM_cell(buffer_input,state)
        with tf.variable_scope('MLP',reuse=False) as scope:
            mlp_input=tf.concat([state_sequences[0],state[1]],axis=1)
            sentence_prediction=self.mlp(input_data=mlp_input, hidden_sizes=self.mlp_neurons)
            predictions.append(sentence_prediction)
        for sentence_idx in xrange(self.sequence_num-1):
            input_sentence_embedding=tf.multiply(tf.matmul(predictions[-1],expand_matrix),sentence_embeddings[sentence_idx])
            with tf.variable_scope('LSTM',reuse=True) as scope:
                _, state=LSTM_cell(input_sentence_embedding,state)
            with tf.variable_scope('MLP',reuse=True) as scope:
                mlp_input=tf.concat([state_sequences[sentence_idx+1],state[1]],axis=1)
                sentence_prediction=self.mlp(input_data=mlp_input, hidden_sizes=self.mlp_neurons)
                predictions.append(sentence_prediction)

        # Generate final prediction and loss
        self.final_prediction=tf.concat(predictions, axis=1)        # of size [self.batch_size, self.sequence_num]
        self.loss= -tf.reduce_mean(tf.multiply(tf.to_float(self.masks), tf.multiply(
            tf.to_float(self.labels), tf.log(self.final_prediction))+tf.multiply(1.0 - tf.to_float(self.labels), tf.log(1.0-self.final_prediction))))

        if self.update_policy['name'].lower() in ['sgd', 'stochastic gradient descent']:
            learning_rate=self.update_policy['learning_rate']
            momentum=0.0 if not self.update_policy.has_key('momentum') else self.update_policy['momentum']
            self.optimizer=tf.train.MomentumOptimizer(learning_rate, momentum)
        elif self.update_policy['name'].lower() in ['adagrad',]:
            learning_rate=self.update_policy['learning_rate']
            initial_accumulator_value=0.1 if not self.update_policy.has_key('initial_accumulator_value') \
                else self.update_policy['initial_accumulator_value']
            self.optimizer=tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value)
        elif self.update_policy['name'].lower() in ['adadelta']:
            learning_rate=self.update_policy['learning_rate']
            rho=0.95 if not self.update_policy.has_key('rho') else self.update_policy['rho']
            epsilon=1e-8 if not self.update_policy.has_key('epsilon') else self.update_policy['epsilon']
            self.optimizer=tf.train.AdadeltaOptimizer(learning_rate, rho, epsilon)
        elif self.update_policy['name'].lower() in ['rms', 'rmsprop']:
            learning_rate=self.update_policy['learning_rate']
            decay=0.9 if not self.update_policy.has_key('decay') else self.update_policy['decay']
            momentum=0.0 if not self.update_policy.has_key('momentum') else self.update_policy['momentum']
            epsilon=1e-10 if not self.update_policy.has_key('epsilon') else self.update_policy['epsilon']
            self.optimizer=tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon)
        else:
            raise ValueError('Unrecognized Optimizer Category: %s'%self.update_policy['name'])
        self.update=self.optimizer.minimize(self.loss)

    def convpool(self, input_data_3d, filter_width, name, stddev=0.02):
        '''
        >>> Construct a convolutional-pooling layer
        >>> input_data: tf.Variable, input data of size [self.batch_size, self.sequence_length, self.embedding_dim]
        >>> filter_width: int, the width of the filter
        >>> name: str, the name of this layer
        >>> stddev: float, the standard derivation of the weight for initialization
        '''
        input_data=tf.expand_dims(tf.reshape(input_data_3d,shape=[self.batch_size,self.sequence_length,self.embedding_dim]),-1)
        with tf.variable_scope(name):
            W=tf.get_variable(name='W',shape=[filter_width,self.embedding_dim,1,self.feature_map],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_output=tf.nn.conv2d(input_data,W,strides=[1,1,1,1],padding='VALID')
            pool_output=tf.nn.max_pool(conv_output,ksize=[1,self.sequence_length-filter_width+1,1,1],
                strides=[1,self.sequence_length-filter_width+1,1,1],padding='VALID')
            return tf.reshape(pool_output,shape=[self.batch_size,self.feature_map])

    def mlp(self, input_data, hidden_sizes, name='mlp', stddev=0.02):
        '''
        >>> Construct a multilayer perceptron model
        >>> input_data: tf.Variable, input data of size [self.batch_size, rnn_size*2]
        >>> hidden_sizes: list<int>, number of neurons in each hidden layer, including the number of neurons in the input layer
        >>> name: str, the name of this model
        >>> stddev: float, the standard derivation of the weight for initialization
        '''
        with tf.variable_scope(name):
            data=input_data
            for idx,neuron_num in enumerate(hidden_sizes[:-1]):
                input_dim=neuron_num
                output_dim=hidden_sizes[idx+1]
                W=tf.get_variable(name='W%d'%idx,shape=[input_dim,output_dim],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                b=tf.get_variable(name='b%d'%idx,shape=[output_dim,],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                data=tf.nn.relu(tf.add(tf.matmul(data,W),b))
            input_dim=hidden_sizes[-1]
            output_dim=1
            W=tf.get_variable(name='W_final',shape=[input_dim,output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b=tf.get_variable(name='b_final',shape=[output_dim,],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
            data=tf.nn.sigmoid(tf.add(tf.matmul(data,W),b))                     # of size [self.batch_size, 1]
            return data

    def train_validate_test_init(self):
        '''
        >>> Initialize the training validation and test phrase
        '''
        self.sess=tf.Session()
        init=tf.global_variables_initializer()
        self.sess.run(init)
        self.decision_func=np.vectorize(lambda x: 1 if x>0.5 else 0)

    def train(self,inputs,masks,labels):
        '''
        >>> Training process on a batch data
        >>> inputs: np.array, of size [self.batch_size, self.sequence_num,self.sequence_length]
        >>> masks: np.array, of size [self.batch_size, self.sequence_num]
        >>> labels: np.array, of size [self.batch_size, self.sequence_num]
        '''
        train_dict={self.inputs:inputs, self.masks:masks, self.labels:labels}
        self.sess.run(self.update,feed_dict=train_dict)
        final_prediction_this_batch, loss_this_batch=self.sess.run([self.final_prediction, self.loss],feed_dict=train_dict)
        final_prediction_this_batch=self.decision_func(final_prediction_this_batch)
        return final_prediction_this_batch, loss_this_batch

    def validate(self,inputs,masks,labels):
        '''
        >>> Validation phrase
        >>> Parameter table is the same as self.train
        '''
        validate_dict={self.inputs:inputs, self.masks:masks, self.labels:labels}
        final_prediction_this_batch, loss_this_batch=self.sess.run([self.final_prediction, self.loss],feed_dict=validate_dict)
        final_prediction_this_batch=self.decision_func(final_prediction_this_batch)
        return final_prediction_this_batch, loss_this_batch

    def test(self,inputs,masks):
        '''
        >>> Test phrase
        >>> Parameter table is almost the same as self.train
        >>> No labels are provided
        '''
        test_dict={self.inputs:inputs, self.masks:masks}
        final_prediction_this_batch=self.sess.run([self.final_prediction],feed_dict=test_dict)
        final_prediction_this_batch=self.decision_func(final_prediction_this_batch)
        return final_prediction_this_batch

    def train_validate_test_end(self):
        '''
        >>> End current training validation and test phrase
        '''
        self.sess.close()
        self.sess=None
