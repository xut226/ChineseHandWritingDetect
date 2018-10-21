# -*- coding :utf-8 -*-
import cv2
from skimage.color import rgb2gray
from chwr import PATH_HOME
from chwr.train import CNNModels
from chwr.DataProcess import Chinese_encoding
from chwr.simpleCNN import SimpleCNN
import matplotlib.pyplot as plt
from chwr.Imageprocess import ImageProcess
__author__ = 'xt'

def Detect_HandWriting(model,net):
    cap = cv2.VideoCapture(0)
    while(1):
        ret,image = cap.read()  #读取一帧图像
        image_gray = rgb2gray(image)    #转换为灰度图像
        imagepro = ImageProcess(image_gray)
        chull = imagepro.findconvexhull(imagepro.image)
        rectangle = imagepro.findBoundingbox(chull)
        if rectangle is not None:
            test_image = imagepro.saveclipImage(image_gray,rectangle)
            xmin,xmax,ymin,ymax = rectangle
        else:
            test_image = None
        # sortedContours,_ = findROI(image)
        # image_ROI,xmin,ymin,xmax,ymax = mergeROI(image,sortedContours)
        # test_image = read_image(PATH_HOME + "\\TestIamges\\testImage.jpg")

        if test_image is not None:
            test_image = imagepro.read_image(PATH_HOME + "TestImages\\testImage.jpg")
            ChineseChar = model.test(test_image,net)
            imagepro.showBBoxwith(image,xmin,ymin,xmax,ymax,ChineseChar)
            print(ChineseChar)
        else:
            cv2.imshow("",image)
            print("there is no char in image!")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__=='__main__':
    imageshape,char_dict,lable_fit = Chinese_encoding(PATH_HOME)   #获取汉字编码
    model = CNNModels(imageshape,char_dict,lable_fit)
    simplenet = SimpleCNN(model.X,model.keep_prob,model.charNum)
    Detect_HandWriting(model,simplenet)

    # image = read_image(PATH_HOME + "TestImages\\testImage.jpg")
    # ChineseChar = model.test(image,simplenet)
    # print(ChineseChar)

