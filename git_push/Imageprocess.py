# -*- coding:utf8 -*-
from PIL import ImageFont,Image,ImageDraw
from selectivesearch import selectivesearch
from skimage import filters
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.morphology import convex_hull_image,dilation
import skimage.morphology as sm
import matplotlib.pyplot as plt
import cv2
import numpy as np

__author__ = 'xutao'

class ImageProcess():
    def __init__(self,image):
        self.image = image
    '''
    凸包找单个文字区域
    '''
    #find convex hull,and return the appropriate boundingbox,2018-10-20
    def findconvexhull(self,image):
        if len(image.shape) == 3:
            image = rgb2gray(image)
        image_sobel = filters.sobel(image)
        image_gaussion = filters.gaussian(image_sobel)
        # threshold = filters.threshold_adaptive(image_gaussion,3,method='gaussian',mode='nearest')
        threshold = filters.threshold_otsu(image_gaussion)
        image_threshold = image_gaussion <= threshold
        image_threshold = np.asarray(image_threshold,dtype=np.uint8)
        m,n = np.shape(image_threshold)
        for i in range(m):
            for j in range(n):
                if image_threshold[i][j] == 0:
                    image_threshold[i][j] = 1
                else:
                    image_threshold[i][j] = 0
        #dilation
        image_dilation = dilation(image_threshold,sm.square(2))
        chull = convex_hull_image(image_dilation)
        return chull

    def findBoundingbox(self,chull):
        contour = find_contours(np.asarray(chull,dtype=np.uint8),0)
        if contour != []:
            ymax,xmax = np.max(contour[0],axis=0)
            ymin,xmin = np.min(contour[0],axis=0)

            #判断矩形框长宽比例是否过大
            if (ymax - ymin) / (xmax - xmin) < 0.2:
                centery = (ymax + ymin) / 2
                ymin = centery - (xmax-xmin) / 2
                ymax = centery + (xmax-xmin) / 2
            elif (xmax - xmin) / (ymax - ymin) < 0.2:
                centery = (xmax + xmin) / 2
                xmin = centery - (ymax-ymin) / 2
                xmax = centery + (ymax-ymin) / 2

            if xmin - 20 > 0 :
                xmin = xmin - 20
            if ymin - 20 > 0:
                ymin = ymin - 20

            return (int(xmin),int(xmax)+20,int(ymin),int(ymax)+20)  #四周留空白余量10
        else:
            return None

    def showBBoxwith(self,image,xmin,ymin,xmax,ymax,text):
        image = cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),4)
        font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
        pilimg= Image.fromarray(np.asarray(image))
        draw = ImageDraw.Draw(pilimg)
        draw.text((xmin + 10,ymin + 10),text,(0,255,0),font=font)
        cv2charimg = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
        cv2.imshow("",cv2charimg)

    def saveclipImage(self,image,rectangle):
        mergedimage = image[rectangle[2]:rectangle[3],rectangle[0]:rectangle[1]]
        cv2.imwrite('TestImages\\testImage.jpg',mergedimage*255)
        return mergedimage

    def read_image(self,path):
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


    '''
    感兴趣区域提取文字
    '''
    def findROI(self,image):
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

    #find ROI using selective search
    def searchROI(path):
        image = cv2.imread(path)
        if image is None:
            return
        img_lbl,regions = selectivesearch.selective_search(image,scale=100,sigma=0.9,min_size=200)
        print(len(regions))
        fig,ax = plt.subplots(figsize=(10,10))
        for reg in regions:
            x,y,w,h = reg['rect']
            rect = cv2.rectangle(image,(x,y ),(x+ w,y+h),(0,255,0),2)
        ax.imshow(rect)
        plt.show()

    def mergeROI(self,image,contours):
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

    '''
    从多个文字图片中分割文字
    '''
    def detect(self):

        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        #形态学
        dilation = self.preprocess(gray)

        #查找和筛选文字区域
        region = self.findTextRegion(dilation)

        # 4. 用绿线画出这些找到的轮廓
        if region is not None:
            for box in region:
                cv2.drawContours(self.image, [box], 0, (0, 255, 0), 2)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", self.image)

    def preprocess(self,image):
        sobel = cv2.Sobel(image,cv2.CV_8U,1,0,ksize = 3)
         # 2. 二值化
        ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

        # 3. 膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

        # 4. 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations = 1)

        # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
        erosion = cv2.erode(dilation, element1, iterations = 1)

        # 6. 再次膨胀，让轮廓明显一些
        dilation2 = cv2.dilate(erosion, element2, iterations = 3)
        return dilation2

    def findTextRegion(self,image):
         region = []

         # 1. 查找轮廓
         _,contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

         # 2. 筛选那些面积小的
         for i in range(len(contours)):
            cnt = contours[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)

            # 轮廓近似，作用很小
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.minAreaRect(cnt)
            # box是四个点的坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算高和宽
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])

            # 筛选那些太细的矩形，留下扁的
            if(height > width * 1.2):
                continue
            region.append(box)
            return region

    def charArea(self,image):

        mser = cv2.MSER_create(_min_area=200)
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        regions,boxes = mser.detectRegions(gray)

        for box in boxes:
            x,y,w,h= box
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("",image)

        cv2.waitKey(0)

