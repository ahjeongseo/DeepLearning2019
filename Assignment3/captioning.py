# This library is used for Assignment3_Part2_ImageCaptioning

# Write your own image captiong code
# You can modify the class structure
# and add additional function needed for image captionging

import tensorflow as tf
import numpy as np


class Captioning():
    def __init__(self):
        self.lr = 0.0001
        self.embedding_dim = 256
        self.units = 512
        self.layers = 2

    def build_model(self,n_words, maxlen, batch_size, training):
        self.img_features = tf.placeholder(dtype=tf.float32,shape = [None, 512])
        self.captions = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.targets = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.keep_prob = 0.8
        
        rnn = tf.contrib.rnn
        
        img_encode = tf.layers.dense(inputs=self.img_features, units=self.embedding_dim,activation=tf.nn.relu)
        
        embedding_layer = tf.get_variable('embedding_layer', shape=[n_words, self.embedding_dim])
        word_embed = tf.nn.embedding_lookup(embedding_layer, self.captions)
        
        cells = []
        with tf.variable_scope("lstm") as l:
            for _ in range(self.layers):
                cell = rnn.LSTMCell(self.units)
                if training:
                    cell = rnn.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
                cells.append(cell)
            
            stacked_lstm = rnn.MultiRNNCell(cells, state_is_tuple=True)
            
                
            self.reset_state = stacked_lstm.zero_state(batch_size,dtype=tf.float32)
            _, self.initial_state = stacked_lstm(img_encode,self.reset_state)
            l.reuse_variables()

            self.outputs, self.final_state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=word_embed,
                                                    sequence_length=tf.constant(16,shape=[batch_size]),
                                                    initial_state=self.initial_state, scope=l)
            self.outputs = tf.reshape(self.outputs,[-1,self.units])
        #print(outputs)
        
        self.logits = tf.layers.dense(inputs=self.outputs, units=n_words)
        self.pred = tf.argmax(tf.nn.softmax(self.logits),1)
        labels = tf.reshape(self.targets, [-1])
        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=self.logits))
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def predict(self):
        captions = None


