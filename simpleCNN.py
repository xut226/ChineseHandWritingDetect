# -*- coding:utf8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

__author__ = 'xt'



class SimpleCNN():
    def __init__(self,images,keep_prob,char_num):
        self.images =images
        self.keep_prob = keep_prob
        self.char_num = char_num
        self.weights = {'conv1':    tf.get_variable(name='conv1',shape=[3,3,1,32],dtype=tf.float32,initializer=xavier_initializer()),
                   'conv2':    tf.get_variable(name='conv2',shape=[3,3,32,64],dtype=tf.float32,initializer=xavier_initializer()),
                   'conv3':    tf.get_variable(name='conv3',shape=[3,3,64,128],dtype=tf.float32,initializer=xavier_initializer()),
                   'fc'   :    tf.get_variable(name='fc',shape=[8192,1024],dtype=tf.float32,initializer=xavier_initializer()),
                   'output':   tf.get_variable(name='output',shape=[1024,self.char_num],dtype=tf.float32,initializer=xavier_initializer())}
        self.biases = {'bias1':     tf.Variable(tf.constant(0.001,shape=[32])),
                  'bias2':     tf.Variable(tf.constant(0.001,shape=[64])),
                  'bias3':     tf.Variable(tf.constant(0.001,shape=[128])),
                  'fc_bias':   tf.Variable(tf.constant(0.001,shape=[1024])),
                  'output':    tf.Variable(tf.constant(0.001,shape=[self.char_num]))}

    def networks(self):
        with tf.name_scope("conv1"):    #output 32 * 32 * 32
            conv_1 = tf.nn.conv2d(input= self.images,filter=self.weights['conv1'],strides=[1,1,1,1],padding='SAME')
            h_relu1 = tf.nn.relu(conv_1 + self.biases['bias1'])
            h_pool1 = tf.nn.max_pool(h_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        with tf.name_scope("conv2"):    #output 16 * 16 * 64
            conv_2 = tf.nn.conv2d(input= h_pool1,filter = self.weights['conv2'],strides=[1,1,1,1],padding='SAME')
            h_relu2 = tf.nn.relu(conv_2 + self.biases['bias2'])
            h_pool2 = tf.nn.max_pool(h_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        with tf.name_scope("conv3"):    #output 8 * 8 * 128
            conv_3 = tf.nn.conv2d(input= h_pool2,filter = self.weights['conv3'],strides=[1,1,1,1],padding='SAME')
            h_relu3 = tf.nn.relu(conv_3 + self.biases['bias3'])
            h_pool3 = tf.nn.max_pool(h_relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        flatten = tf.contrib.layers.flatten(h_pool3)

        with tf.name_scope("fc1") :
            h_fc = tf.matmul(flatten,self.weights['fc'])
            h_fc = tf.nn.tanh(tf.add(h_fc,self.biases['fc_bias']))
            h_fc = tf.nn.dropout(h_fc,self.keep_prob)
        with tf.name_scope("fc2"):
            h_fc2 = tf.matmul(h_fc,self.weights['output'])
            logits = tf.nn.dropout(h_fc2,self.keep_prob)


        tf.summary.histogram("weight1",self.weights['conv1'])
        tf.summary.histogram("activation1",h_relu1)
        tf.summary.histogram("weight3",self.weights['conv3'])
        tf.summary.histogram("activation3",h_relu3)
        tf.summary.image("conv1",conv_1[:,:,:,0:3])
        tf.summary.image("conv3",conv_3[:,:,:,0:3])

        return logits,conv_1,conv_3