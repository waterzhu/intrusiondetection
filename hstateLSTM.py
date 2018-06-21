
import os
import random
import time
import tensorflow as tf
import numpy as np
import math

class HIDSNet(object):
    def __init__(self,
                 dropout,
                 classes,
                 flow_length,
                 batch_size):
        self.class_num = classes
        self.dropout = dropout
        self.size = 64  #size of hidden nodes in LSTM
        self.batchsize = batch_size
        self.fcsize = 64*5 #size of hidden nodes in FC
        self.flow_length = flow_length  #the length of packet of flows
        self.x_flow = tf.placeholder(tf.float32, shape=[None, self.flow_length, 78], name = "input_flow")   #input tensor [batch, flowlength, feature_length]
        self.y = tf.placeholder(tf.int32, shape=[None], name = "flow_class")
        self.attention = tf.constant([0.1, 0.1, 0.1, 0.1, 0.6])
        self.creat_model()

    def creat_model(self):
        state_out = self.hstateLSTM()
        self.loss, self.acc, self.pre_y = self.flow_classification(state_out)

    def hstateLSTM(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.size)
        out = []
        with tf.variable_scope("Simplelstm"):
            state = cell.zero_state(self.batchsize, dtype = tf.float32)
            for flow in range(self.flow_length):
                if flow > 0:
                    tf.get_variable_scope().reuse_variables()
                output, (c_state, h_state) = cell(self.x_flow[:, flow, :], state)
                if flow < self.flow_length-1:
                    out.append(h_state)
                else:
                    out.append(h_state*10)
#            print(out)
#            out = tf.matmul(out, self.attention)
            out = tf.concat(out, 1)

#            out = output
#            print(out)
        return out

    def flow_classification(self, out):
        with tf.name_scope('fc'):
            out_dropout = tf.nn.dropout(out, self.dropout)
            W = tf.get_variable("full", [self.fcsize, self.class_num])
            bias = tf.Variable(tf.constant(0.1, shape=[self.class_num]), dtype=tf.float32)
            class_output = tf.nn.softmax(tf.matmul(out_dropout, W) + bias)

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
