# -*- coding:utf-8 -*-

import sys
import os
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import slim
from tensorflow.python.training.summary_io import SummaryWriter
from chwr import PATH_HOME
from chwr.DataProcess import generate_data_and_label, Chinese_encoding
from tensorflow.python import debug as tfbug
from tensorflow.contrib.layers import variance_scaling_initializer
import matplotlib.pyplot as plt
from chwr.VGGnet import VGGnet
from chwr.simpleCNN import SimpleCNN

cur_path = sys.path[0]

__author__ = 'xt'

flags = tf.flags

flags.DEFINE_string("ckpt_dir","..\\chwr\\ckpt_dir\\","save the model: sess")
flags.DEFINE_integer("charNum",3755,"the total num of Chinese characters classes")
flags.DEFINE_string("logs","..\\chwr\\logs\\","tensorboard summary")
flags.DEFINE_integer("batch_size",32,"trainning batch size")


FLAGS = flags.FLAGS

class CNNModels:
    def __init__(self,imageshapelist,char_dict,lable_fit):
        if len(imageshapelist) > 0:
            self.image_size = np.asarray(imageshapelist,dtype=np.uint8)
        self.char_dict = char_dict      #获取汉字编码，字典
        self.charNum = len(char_dict)   #汉字类别个数
        self.lable_fit = lable_fit      #二值化汉字编码

        self.filenames = []             #文件列表
        self.labels = []                #标签
        data_dir = PATH_HOME + "TrainingData"

        for root,sub_folders,file_list in os.walk(data_dir):
            self.filenames  =  file_list
        self.labels = [int(((filename.split("+")[1]).split("."))[0]) for filename in self.filenames]

        self.filenames  = [data_dir + "\\"  + filename for filename in self.filenames]

        # self.X = tf.placeholder("float32",shape=(None,self.image_size[0],self.image_size[1],1))

        self.X = tf.placeholder("float32",shape=(None,64,64,1))
        # self.Label = tf.placeholder(tf.int64,shape=(None,self.charNum))
        self.Label = tf.placeholder(tf.int64,shape=(None))

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')



    def Input_recordfile_list(self,recordfile_dir):
        for root,sub_folder,filenames in os.walk(recordfile_dir):
            recordfile_list = [root +"\\" + filename for filename in filenames]

        filename_queue = tf.train.string_input_producer(recordfile_list,shuffle=False)
        reader = tf.TFRecordReader()
        _,serialized_examples = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_examples,
            features={
                'label':tf.FixedLenFeature([],tf.int64),
                'image':tf.FixedLenFeature([],tf.string),
                }
        )

        image = tf.decode_raw(features['image'],tf.uint8)
        image = tf.cast(image,dtype=tf.float32)
        image = tf.reshape(image,[64,64,1])
        image = self.data_augmentation(image)

        label = tf.cast(features['label'],tf.int64)

        image_tensor,label_tensor = tf.train.shuffle_batch([image,label],batch_size=FLAGS.batch_size,capacity=50000,min_after_dequeue=10000)
        return image_tensor,label_tensor


    def data_augmentation(self,images):
        # images = tf.image.random_flip_up_down(images)
        images = tf.image.random_brightness(images, max_delta=0.3)
        images = tf.image.random_contrast(images, 0.8, 1.2)
        return images


    def CostFunction(self,inferece):
        y_pred,conv1,conv3 = inferece.networks()
        y_log = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Label,logits=y_pred)

        #cost
        cost = tf.reduce_mean(y_log)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0), trainable=False)

        rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=100, decay_rate=0.999995, staircase=True)
        #optimizer
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost,global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost,global_step=global_step)
        accuracy_equal = tf.equal(tf.arg_max(y_pred,1),self.Label)
        accuracy = tf.reduce_mean(tf.cast(accuracy_equal,tf.float32))

        tf.summary.scalar('cost',cost)
        tf.summary.scalar('accuracy',accuracy)

        return optimizer,y_pred,accuracy,cost,global_step,conv1,conv3

    def test(self,image,net):
        image_tensor = tf.cast(image,tf.float32)
        image_tensor = tf.reshape(image,[1,64,64,1])
        path = PATH_HOME + "\\ckpt_dir"
        saver = tf.train.Saver()
        y_pred, _, _ = net.networks()
        logits = tf.nn.softmax(y_pred)
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                predicts = sess.run([logits],feed_dict={
                    self.X:image,
                    self.keep_prob:1
                })
        value = np.argmax(predicts)
        for key in self.char_dict:
            if self.char_dict[key] == value:
                ChineseChar = key
        return ChineseChar

    def pre_trian(self,net,pred_image=None):
        '''
        :param pred_image: 实时输入的手写待识别汉字图片
        :return:
        '''
        if not os.path.exists(FLAGS.ckpt_dir):
            os.makedirs(FLAGS.ckpt_dir)
        # global_step = tf.Variable(0,name='step',trainable=False)
        # f_log = open(ckpt_dir + "\\log.txt",'w')

        with tf.Session() as sess:
            optimizer,predict,accuracy,cost,global_step,conv1,conv3 = self.CostFunction(net)
            count = 0;epoch = 0

            train_batch_images_tensor,train_batch_labels_tensor = self.Input_recordfile_list(PATH_HOME+"\\tfrecord\\train")
            test_batch_images_tensor,test_batch_labels_tensor = self.Input_recordfile_list(PATH_HOME+"\\tfrecord\\test")
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess,coord=coord)
            saver = tf.train.Saver(max_to_keep=3)   #默认保存5个，生成HCCR.ckpt-250...
            summary_writer = SummaryWriter(FLAGS.logs,sess.graph)
            merge = tf.summary.merge_all()
            model_file = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
            if  model_file:
                saver.restore(sess,model_file)
                count = int(model_file.split('-')[1])

            try:
                while not coord.should_stop():
                    start_time = time.time()
                    train_batch_images,train_batch_labels = sess.run([train_batch_images_tensor,train_batch_labels_tensor])
                    train_batch_images = train_batch_images / 256

                    _,predicts,train_accuracy,costs,step, summary,conv3image = sess.run([
                        optimizer,predict,accuracy,cost,global_step,merge,conv3],feed_dict={
                        self.X:train_batch_images,
                        self.Label:train_batch_labels,
                        self.keep_prob:0.8

                    })
                    count += 1
                    end_time =time.time()
                    gap_times = end_time - start_time
                    str_step = "count:%d,step: %d,time: %d, train_accuracy: %f, cost: %f" % (count,step,gap_times,train_accuracy,costs)
                    print(str_step)

                    with open(FLAGS.ckpt_dir + "\\log.txt",'a') as f_log:
                        f_log.write(str_step + "\n")

                    if count % 50== 0:
                        saver.save(sess,FLAGS.ckpt_dir + 'chwr.ckpt',global_step=count)

                    if count % 50 == 0:
                        test_images_batch, test_labels_batch = sess.run([test_batch_images_tensor, test_batch_labels_tensor])
                        feed_dict = {self.X: test_images_batch,
                                     self.Label: test_labels_batch,
                                     self.keep_prob: 1.0}
                        accuracy_test = sess.run(
                            [accuracy],
                            feed_dict=feed_dict)

                        print('===============Eval a batch=======================')
                        print('the step {0} test accuracy: {1}'
                                    .format(step, accuracy_test))
                        print('===============Eval a batch=======================')
                    summary_writer.add_summary(summary,global_step=count)

            except tf.errors.OutOfRangeError:
                print("train finished")
            finally:
                coord.request_stop()
            coord.join(threads)

    def train(self,net):
        self.pre_trian(net)


    def conv2image(self,conv,step):
        plt.figure()

        m,n,l,k = np.shape(conv)
        # for i in range(k):
            # plt.subplot(n,l,i+1)
        plt.imshow(conv[0,:,:,0])
        plt.savefig('scatterimage\\' +'3_' +str(step) + '.jpg')
        plt.close()

if __name__ == '__main__':
    imageshape,char_dict,lable_fit = Chinese_encoding(PATH_HOME)   #汉字编码
    model = CNNModels(imageshape,char_dict,lable_fit)
    simplenet = SimpleCNN(model.X,model.keep_prob,model.charNum)
    # vggget = VGGnet(model.X,model.keep_prob,model.charNum)
    model.train(simplenet)



