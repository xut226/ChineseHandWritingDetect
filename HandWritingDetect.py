# -*- coding :utf-8 -*-
import PIL
from PIL import Image,ImageDraw,ImageFont
import cv2
from numba import uint8
import numpy as np
from numpy.matlib import ones
from chwr import PATH_HOME
from chwr.train import CNNModels
from chwr.DataProcess import Chinese_encoding
from chwr.simpleCNN import SimpleCNN
import matplotlib.pyplot as plt
from chwr.Imageprocess import ImageProcess
import  PIL
from PIL import ImageOps
__author__ = 'xt'

def Detect_HandWriting(model,net):
    cap = cv2.VideoCapture(0)
    while(1):
        ret,image = cap.read()  #读取一帧图像
        # imageprocess = ImageProcess(image)
        # imageprocess.detect()

        sortedContours,_ = findROI(image)
        image_ROI,xmin,ymin,xmax,ymax = mergeROI(image,sortedContours)
        test_image = read_image(PATH_HOME + "\\TestIamges\\testImage.jpg")
        if test_image is not None:
            ChineseChar = model.test(test_image,net)
            print(ChineseChar)
            showBBoxwith(image,xmin,ymin,xmax,ymax,ChineseChar)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def findROI(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    m,n = image_gray.shape

    #sobel算子
    gradX = cv2.Sobel(image_gray,ddepth=cv2.CV_32F,dx=1,dy=0)
    gradY = cv2.Sobel(image_gray,ddepth=cv2.CV_32F,dx=0,dy=1)
    gradient = cv2.subtract(gradX,gradY)
    gradient = cv2.convertScaleAbs(gradient)
    #均值滤波
    image_blur = cv2.GaussianBlur(gradient,(5,5),0)   #均值滤波
    #二值化处理
    ret,thresh = cv2.threshold(image_blur,13,255,cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   #构造一个正方形内核
    # 执行图像形态学
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    #轮廓,
    image_contour,contours,hierarchy = cv2.findContours(closed.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sortedContours = sorted(contours,key=cv2.contourArea,reverse=True)

    if sortedContours is not None:
        c1 = sortedContours[0]
        c2 = sortedContours[1]
        rect1 = cv2.minAreaRect(c1)
        box1 = np.int0(cv2.boxPoints(rect1))
        rect2 = cv2.minAreaRect(c2)
        box2 = np.int0(cv2.boxPoints(rect2))
        draw_img = cv2.drawContours(image.copy(),[box1],-1,(0,0,255),3)
        draw_img = cv2.drawContours(draw_img.copy(),[box2],-1,(0,0,255),3)

        cv2.imshow("image",draw_img)
        return sortedContours,draw_img
    else:
        return None,None

def mergeROI(image,contours):
    height_img,width_img,_ = np.shape(image)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1.append( min(Xs) )
        x2.append( max(Xs) )
        y1.append( min(Ys) )
        y2.append( max(Ys) )

    if x1 is None:
        return
    x1.sort()
    x2.sort()
    y1.sort()
    y2.sort()

    x_min = x1[0]
    x_max = x1[-1]
    y_min = y1[0]
    y_max = y2[-1]
    width = (x_max - x_min)
    height = (y_max - y_min)

    if x_min - 20 > 0 and width + 20 < width_img:
        x_min -= 20
        width += int(width / 10)
    if y_min - 20 > 0 and height + 20 < height_img:
        y_min -= 20
        height += int(height / 10)
    mergedimage = image[y_min:y_min+height,x_min:x_min+width]

    cv2.imwrite('TestIamges\\testImage.jpg',mergedimage)

    return mergedimage,x_min,y_min,x_max,y_max

def showBBoxwith(image,xmin,ymin,xmax,ymax,text):
    image = cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),4)
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")

    pilimg= Image.fromarray(np.asarray(image))
    draw = ImageDraw.Draw(pilimg)
    draw.text((xmin + 30,ymax- 30),text,(0,255,0),font=font)
    # cv2.putText(image,text,(xmin + 20,ymax-20),font,1.2,(0,255,0),2)
    cv2charimg = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
    cv2.imshow("",cv2charimg)


def read_image(path):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if image is None:
        return
    reshaped_image = cv2.resize(image,(64,64))
    ret,thresh = cv2.threshold(reshaped_image,100,255,cv2.THRESH_BINARY)
    m,n = np.shape(thresh)
    for i in range(m):
        for j in range(n):
            thresh[i,j] = 255 - thresh[i,j]

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    dilation = cv2.dilate(thresh, element, iterations = 1)
    for i in range(m):
        for j in range(n):
            dilation[i,j] = 255 - dilation[i,j]
    cv2.imwrite("ResizedIamge.jpg",dilation)
    reshaped_image = np.reshape(dilation,(1,64,64,1))
    return reshaped_image

if __name__=='__main__':

    imageshape,char_dict,lable_fit = Chinese_encoding(PATH_HOME)   #汉字编码
    model = CNNModels(imageshape,char_dict,lable_fit)
    simplenet = SimpleCNN(model.X,model.keep_prob,model.charNum)
    Detect_HandWriting(model,simplenet)
    # image = read_image(PATH_HOME + "\\testImage.jpg")
    # ChineseChar = model.test(image,simplenet)
    # print(ChineseChar)
