# -*- coding:utf8 -*-


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, xavier_initializer

__author__ = 'xutao'

class VGGnet:

    def __init__(self,images,keep_prob,char_num):
        self.images =images
        self.keep_prob = keep_prob
        self.char_num = char_num

    #定义卷积操作
    def conv_op(self,input_op,name,kh,kw,n_out,dh,dw,p):
        """
        :param input_op:输入tensor
        :param name:名称
        :param kh:卷积核高
        :param kw:卷积核宽
        :param n_out:卷积核数量，即输出通道数
        :param dh:步长的高
        :param dw:步长宽
        :param p:参数列表
        :return:
        """
        n_in = input_op.get_shape()[-1].value   #输入通道数

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope+'w',shape=[kh,kw,n_in,n_out],dtype=tf.float32,
                    initializer = xavier_initializer(),    #xavier 开平方根
                                     )
        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.001,shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val,trainable=True,name='b')
        z = tf.nn.bias_add(conv,bias_init_val)
        activation = tf.nn.relu(z)
        p += [kernel,biases]

        tf.summary.histogram("weight_" + name,kernel)
        tf.summary.image(name,conv[:,:,:,0:3])
        return activation

    def mpool_op(self,input_op,name,kh,kw,dh,dw):
        '''
        :param input_op:
        :param name:
        :param kh:
        :param kw:
        :param dh:
        :param dw:
        :return:
        '''
        return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

    def fc_op(self,input_op,name,n_out,p):
        '''
        :param input_op:
        :param name:
        :param n_out:
        :param p:
        :return:
        '''
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope+ 'w',shape=[n_in,n_out],dtype=tf.float32,
                                     initializer=xavier_initializer())
            biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
            # activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)

            activation = tf.nn.tanh(tf.add(tf.matmul(input_op,kernel),biases),name=scope)
            p += [kernel,biases]
        return activation


    def networks(self):
        p = []
        conv1_1 = self.conv_op(self.images,name="conv1_1",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
        conv1_2 = self.conv_op(conv1_1,name="conv1_2",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
        pool1 = self.mpool_op(conv1_2,name='pool1',kh=2,kw=2,dh=2,dw=2) #output: 32*32

        conv2_1 = self.conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
        conv2_2 = self.conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
        pool2 = self.mpool_op(conv2_2,name='pool2',kh=2,kw=2,dh=2,dw=2) #output: 16*16

        conv3_1 = self.conv_op(pool2,name='conv3_1',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
        conv3_2 = self.conv_op(conv3_1,name='conv3_2',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
        pool3 = self.mpool_op(conv3_2,name='pool3',kh=2,kw=2,dh=2,dw=2) #output: 8*8

        conv4_1 = self.conv_op(pool3,name='conv4_1',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
        conv4_2 = self.conv_op(conv4_1,name='conv4_2',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
        conv4_3 = self.conv_op(conv4_2,name='conv4_3',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
        pool4 = self.mpool_op(conv4_3,name='pool4',kh=2,kw=2,dh=2,dw=2)

        # conv5_1 = self.conv_op(pool4,name='conv5_1',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
        # conv5_2 = self.conv_op(conv5_1,name='conv5_2',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
        # conv5_3 = self.conv_op(conv5_2,name='conv5_3',kh=3,kw=3,n_out=512,dh=1,dw=1,p=p)
        # pool5 = self.mpool_op(conv5_3,name='pool5',kh=2,kw=2,dh=2,dw=2)


        shp = pool4.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value

        resh1 = tf.reshape(pool4,[-1,flattened_shape],name='resh1')

        fc6 = self.fc_op(resh1,'fc6',n_out=4096,p=p)
        fc6_drop = tf.nn.dropout(fc6,self.keep_prob,name='fc6_drop')

        fc7 = self.fc_op(fc6_drop,name='fc7',n_out=1024,p=p)
        fc7_drop = tf.nn.dropout(fc7,self.keep_prob,name='fc7_drop')

        # fc8 = self.fc_op(fc7_drop,name='fc8',n_out=1024,p=p)
        # softmax = tf.nn.softmax(fc8)
        # predictions = tf.argmax(softmax,1)

        weight_output = tf.Variable(tf.truncated_normal([1024,self.char_num],stddev=0.001))
        h_output = tf.matmul(fc7,weight_output)

        return h_output,conv1_1,conv4_3

    # def loss(self,h_output):