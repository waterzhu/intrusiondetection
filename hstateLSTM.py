
import os
import random
import time
import tensorflow as tf
import numpy as np
import math

class HIDSNet(object):
    def __init__(self,
		 hidden_node,
		 fc_middle,
		 attentions,
                 dropout,
                 classes,
                 flow_length,
                 batch_size):
        self.class_num = classes
        self.middle = fc_middle
        self.dropout = dropout
        self.size = hidden_node  #size of hidden nodes in LSTM
        self.batchsize = batch_size
	self.attentions = attentions
        self.fcsize = self.size*flow_length #size of hidden nodes in FC
        self.flow_length = flow_length  #the length of packet of flows
        self.x_flow = tf.placeholder(tf.float32, shape=[None, self.flow_length, 78], name = "input_flow")   #input tensor [batch, flowlength, feature_length]
        self.y = tf.placeholder(tf.int32, shape=[None], name = "flow_class")
#        self.attention = tf.constant([0.1, 0.1, 0.1, 0.1, 0.6])
        self.creat_model()

    def creat_model(self):
        state_out = self.hstateLSTM()
        attention_out, alphas = self.attention(state_out, 50, return_alphas=True)
        self.loss, self.acc, self.pre_y = self.flow_classification(attention_out)

    def hstateLSTM(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.size)
        out = []
        with tf.variable_scope("hiddenSimplelstm"):
            state = cell.zero_state(self.batchsize, dtype = tf.float32)
            for flow in range(self.flow_length):
                if flow > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = cell(self.x_flow[:, flow, :], state)
#                if flow < self.flow_length-1:
                out.append(state[1])
#                else:
#                    out.append(state[1]*self.attentions)
#            print(out)
#            out = tf.matmul(out, self.attention)
#            out1 = tf.concat(out, 1)

#            out = output
#	    print(sess.run(tf.shape(out1)))
#            print(tf.shape(out))
#	    print('haha')
        return out



    def attention(self, inputs, attention_size, time_major=True, return_alphas=False):
#        if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
#            inputs = tf.concat(inputs, 2)
        if time_major:
        # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])
        hidden_size = self.size  # D value - hidden size of the RNN layer
#	print(inputs.get_shape())
    # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

#	print(w_omega.get_shape())
        with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = (tf.tensordot(inputs, w_omega,1) + b_omega)
	    print(v.get_shape())
	    v = tf.tanh(v)
	print(v.get_shape())
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
#	print(vu.get_shape())
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
#	print(alphas.get_shape())
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = inputs * tf.expand_dims(alphas, -1)
#	print(tf.shape(output))
        if not return_alphas:
            return output
        else:
            return output, alphas





    def flow_classification(self, out):
	out = tf.reshape(out, shape = [self.batchsize,self.flow_length*self.size])
#	print(out.get_shape())
#	out = tf.concat(out, 1)
#	print(out.get_shape())
        with tf.name_scope('fc'):
            out_dropout = tf.nn.dropout(out, self.dropout)
	    if self.middle == 0:
            	 W = tf.get_variable("full", [self.fcsize, self.class_num])
           	 bias = tf.Variable(tf.constant(0.1, shape=[self.class_num]), dtype=tf.float32)
             	 class_output = tf.nn.softmax(tf.matmul(out_dropout, W) + bias)
	    else:
            	 W = tf.get_variable("full", [self.fcsize, self.middle])
           	 bias = tf.Variable(tf.constant(0.1, shape=[self.middle]), dtype=tf.float32)
             	 middle_output = tf.nn.softmax(tf.matmul(out_dropout, W) + bias)
	         W2 = tf.get_variable("full2", [self.middle, self.class_num])
                 bias2 = tf.Variable(tf.constant(0.1, shape=[self.class_num]), dtype=tf.float32)
                 class_output = tf.nn.softmax(tf.matmul(middle_output, W2) + bias2)

        labels = tf.one_hot(indices = self.y, depth = self.class_num)
        logits = class_output
        pre_y = tf.reshape(tf.argmax(logits, 1), shape=[1, self.batchsize])

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                  name="loss")
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1)), tf.float32),
                name="accuracy")

        print("Model DONE")
        return loss, accuracy, pre_y
