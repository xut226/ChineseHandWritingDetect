# -*- coding: utf-8 -*-
import csv
import cv2
import os
import re
import numpy as np
from PIL import Image
import struct
import binascii

from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import tensorflow as tf
from chwr import PATH_HOME

__author__ = 'xt'


#1.加载label标签(char_dict)，如果没有则先建立标签字典数据
def load_char_dict(Path_home,file='trianingdata'):
    if file == 'trianingdata':
        filenames = [Path_home + "\\train\\%04d-c.gnt" % i for i in range(1001,1240,1)]
    if file == 'testdata':
        filenames = [Path_home + "\\test\\%04d-c.gnt" % i for i in range(1241,1301,1)]

    char_set = set()
    if os.path.exists(Path_home +"char_dict\\") and os.path.exists(Path_home +"char_dict\\char_dict.txt"):
        fp_char = open(Path_home +"char_dict\\char_dict.txt",'r')
        char_dict = eval( fp_char.read() )  #eval 读取为字典格式
        return char_dict
    else:
        if not os.path.exists(Path_home +"char_dict\\"):
            os.mkdir(Path_home +"char_dict\\")
        for filename in filenames:
            # path = os.path.join(Path,filename)
            f = open(filename,'rb')
            while(True):
                header_size = 10
                header = np.fromfile(f,dtype='uint8',count=10)
                if not header.size:
                    break
                sample_size = header[0] + (header[1]<<8) +(header[2]<<16) + (header[3] <<24)
                tag_code = header[5] + (header[4] << 8)
                width = header[6] + (header[7]<<8)
                height = header[8] + (header[9]<<8)
                if header_size + width*height != sample_size:
                    break
                tagcode_unicode = struct.pack('>H',tag_code).decode('gb2312')
                image = np.fromfile(f,dtype='uint8',count=width*height).reshape((height,width))
                char_set.add(tagcode_unicode)
        char_list = list(char_set)
        char_dict = dict(zip(sorted(char_list),range(len(char_list))))

        with open(Path_home +"char_dict\\char_dict.txt",'w') as f_chardict:
            f_chardict.write(str(char_dict))
        return char_dict

#直接从.gnt数据加载图片
def load_data(Path,file='trainingdata'):
    filenames = [Path + "\\train\\%04d-c.gnt" % i for i in range(1001,1240,1)]
    if file == 'trianingdata':
        filenames = [Path + "\\train\\%04d-c.gnt" % i for i in range(1001,1240,1)]
    if file == 'testdata':
        filenames = [Path + "\\test\\%04d-c.gnt" % i for i in range(1241,1301,1)]
    paths = []
    Images = []
    path = Path
    for filename in filenames:
        # path = os.path.join(Path,filename)
        f = open(filename,'rb')
        while(True):
            header_size = 10
            header = np.fromfile(f,dtype='uint8',count=10)
            if not header.size:
                break
            sample_size = header[0] + (header[1]<<8) +(header[2]<<16) + (header[3] <<24)
            tag_code = header[5] + (header[4] << 8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            tagcode_unicode = struct.pack('>H',tag_code).decode('gb2312')
            image = np.fromfile(f,dtype='uint8',count=width*height).reshape((height,width))
            yield image,tagcode_unicode

#3.保存图片到本地
def save_image(Path):
    count = 0
    fp_char = open(PATH_HOME + "\\Chinese_encoding_data\\char_dict.txt",'r')
    char_dict = eval( fp_char.read() )  #eval 读取为字典格式

    for image,tagcode_unicode in load_data(Path):
        count += 1
        filename = "TrainingData\\"+str(count) + "+" + str(char_dict[tagcode_unicode]) + '.jpg'
        height,width = image.shape
        Img = Image.new('RGB',(width,height))
        image_array = Img.load()
        for  i in range(height):
            for j in range(width):
                pixel = image[i][j]
                image_array[j,i] = (pixel,pixel,pixel)

        if(os.path.exists(Path + "TrainingData")):
            filename = Path + filename
            # imagearray = cv2.resize(np.asarray(image_array,dtype=np.uint8),(80,80))
            # cv2.imwrite(filename,Img)
            pil_im = Image.fromarray(np.uint8(Img))
            im = pil_im.resize((64,64))
            im.save(filename)
        else:
            os.mkdir(Path + "TrainingData")
            filename = Path + filename
            # image_array = cv2.resize(np.asarray(image_array,dtype=np.uint8),(80,80))
            # cv2.imwrite(filename,Img)
            pil_im = Image.fromarray(np.uint8(Img))
            im = pil_im.resize((64,64))
            im.save(filename)

        chstr = binascii.b2a_hex(tagcode_unicode.encode('utf-8'))
        image_array = image_array

#4.将汉字图像编码
def Chinese_encoding(Path):
    char_set = set()
    char_dict = {}
    Labels = []
    imageshape_set = set()
    imageshape = np.zeros(2)
    path_save = Path  + "\\Chinese_encoding_data"
    if os.path.exists(path_save) and os.path.exists(path_save+"\\char_dict.txt"):
        for root,dirs,files in os.walk(path_save):
            for file in files:
                if file == "char_dict.txt":
                    fp_char = open(path_save + "\\char_dict.txt",'r')
                    char_dict = eval( fp_char.read() )  #eval 读取为字典格式

                if file == "imageshape.txt":
                    fp_imageshape = open(path_save + "\\imageshape.txt")
                    a = re.findall('\d+',fp_imageshape.read())
                    imageshape = np.asarray(a)
    else:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        path_image = Path+"save_image_shaped\\"
        for root,dirs,files in os.walk(path_image):
            for file in files:
                image = Image.open(path_image + file).convert('L')
                im = Image.fromarray(np.uint8(image))
                pil_im = im.resize((64,64))
                # image = np.asarray(image)
                m,n = np.shape(pil_im)
                imageshape_set.add((m,n))
                if len(file.split('+')) > 1:
                    chinese = file.split('+')[1].rstrip('.')[0]
                if chinese is not None:
                    char_set.add(chinese)
        char_list = list(char_set)
        char_dict = dict(zip(sorted(char_list),range(len(char_list))))
        imageshape = np.array(list(imageshape_set))
        with open(path_save+'\\char_dict.txt',mode='w') as f1:
            f1.write(str(char_dict))
        with open(path_save + "\\imageshape.txt",mode='w') as f2:
            f2.write(str(imageshape))

    fit_Label = label_binarizer(char_dict)

    return imageshape,char_dict,fit_Label #返回图像shape和汉字编码,及二值化标签属性


#标签二值化
def label_binarizer(char_dict):
    #标签二值化
    fit_Label = LabelBinarizer()
    BiLabel = []
    for key in char_dict.keys():
        BiLabel.append(char_dict[key])
    fit_Label.fit(np.asarray(BiLabel,dtype=np.int32))
    # y = []
    # a = fit_Label.transform([12])
    # y.append(char_dict['啊'])
    # a = fit_Label.transform(y)
    return fit_Label

#将对应的二值化label转换为汉字
def biLabel_to_Chinese(label,label_fit,char_dict):
    encoding = label_fit.inverse_transform(label)
    for char,encode in char_dict.items():
        if encode == encoding:
            return  char

#generate batch size of (data, label)
def generate_data_and_label(path,char_dict,label_fit,batch_size=128):
    count = 0
    path_image = path + 'save_image_shaped\\'
    for root,dirs,files in os.walk(path_image):
        batch_Image = []
        batch_label = []
        for file in files:
            image = Image.open(path_image + file).convert('L')
            im = Image.fromarray(np.uint8(image))

            resize_im = np.array(im.resize((64,64)))
            resize_im = np.reshape(resize_im,(64,64,1))
            batch_Image.append(np.asarray(resize_im,dtype=np.float32) / 255)
            label = file.split('+')[1].rstrip('.')[0]
            batch_label.append(char_dict[label])
            count += 1
            # if count % batch_size == 0:
            #     yield np.asarray(batch_Image),label_fit.transform(batch_label)
            #     batch_Image = []
            #     batch_label = []
    return np.asarray(batch_Image),label_fit.transform(batch_label) #这种方式需要运算、存储的数据太多

#获取图像的尺寸
def get_image_size(image):
    size = np.shape(image)
    return size

def _int64_featrue_(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
def _float_feature_(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))
def _byte_feature_(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


#2.保存为TFRecord
def write2tfrecord(path_home,char_dict,datatype = 'trainingdata'):
    count = 0
    writer = None
    resource_path = path_home + "\\Chinese_handwriting_database"
    if not os.path.exists(path_home + "tfrecord"):
            os.mkdir(path_home + "tfrecord")

    for image,tagcode_unicode in load_data(resource_path,datatype):

        if count % 10000 == 0:
            recordCnt = count // 10000
            if writer is not None:
                writer.close()
            if datatype == 'trainingdata':
                writer = tf.python_io.TFRecordWriter(path_home+"tfrecord\\train\\"+"ChwtrainImage" + str(recordCnt) + ".tfrecords")
            if datatype == 'testdata':
                writer = tf.python_io.TFRecordWriter(path_home+"tfrecord\\test\\"+"ChwtestImage" + str(recordCnt) + ".tfrecords")
        filename = "tfrecord\\"+str(count) + "+" + str(char_dict[tagcode_unicode]) + '.jpg'
        height,width = image.shape
        Img = Image.new('RGB',(width,height))
        image_array = Img.load()
        for  i in range(height):
            for j in range(width):
                pixel = image[i][j]
                image_array[j,i] = (pixel,pixel,pixel)
        resized_img = Img.resize((64,64))
        singlechannel_img = resized_img.convert('L')
        # singlechannel_img.show(tagcode_unicode)

        image_raw = singlechannel_img.tobytes()
        label = char_dict[tagcode_unicode]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_featrue_(label),
            'image': _byte_feature_(image_raw),
            }))

        writer.write(example.SerializeToString())
        if count % 100 == 0:
            print("write the %d record" % count)
        count += 1


if __name__=='__main__':
    # SaveImage(PATH_HOME)
    char_dict = load_char_dict(PATH_HOME)
    write2tfrecord(PATH_HOME,char_dict,'trainingdata')
    write2tfrecord(PATH_HOME,char_dict,'testdata')




