from __future__ import print_function, division
import xml
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from lxml import etree, objectify


try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

OUTLIER=1
NOT_OUTLIER=0

def outliterCheck(img, X, i):
    RANGE_SIZE = 64
    flag = 0

    for pad_x in range(int(-RANGE_SIZE / 2), int(RANGE_SIZE / 2)+1):
        for pad_y in range(int(-RANGE_SIZE / 2), int(RANGE_SIZE / 2)+1):
            if (pad_x is not 0) and (pad_y is not 0) and (X[i][1] + pad_x)>0.0 and (X[i][0] + pad_y)>0.0 and (X[i][1] + pad_x)<img.shape[0] and (X[i][0] + pad_y)<img.shape[1]:
                if img[int(X[i][1] + pad_x), int(X[i][0] + pad_y)] == 1.0:
                    flag = 1

    if flag == 0:
        return OUTLIER

    else:
        return  NOT_OUTLIER


list = glob.glob('/home/lee/Documents/Dataset_to_VOC_converter-master/train2/*.xml')
#list = glob.glob('./*.xml')
for annotationPath in list:
    print(annotationPath )
    tree = ET.parse(annotationPath)
    root = tree.getroot()
    annotationObj=root.find('annotation')
    objects = root.findall('object')
    size=root.find('size')
    image_width=int(size.find('width').text)
    image_height = int(size.find('height').text)
    X=[]
    Z=[]
    for i in range(len(objects)):
        bndboxObj = objects[i].find('bndbox')
        xmin = int(bndboxObj.find('xmin').text)
        ymin = int(bndboxObj.find('ymin').text)
        xmax = int(bndboxObj.find('xmax').text)
        ymax = int(bndboxObj.find('ymax').text)
        w=(xmax-xmin)
        h=(ymax-ymin)
        if w<32 and h<32:
            cx=(xmax+xmin)/2
            cy=(ymax+ymin)/2
            X.append([cx,cy])
    if len(X) <1:
        continue
    img=np.zeros((image_height,image_width))
    for i in range(len(X)):
        img[int(X[i][1]), int(X[i][0])]=1.0

    ret = 1
    #print('before : ', len(X))
    for i in range(len(X)):
        ret=outliterCheck(img, X, i)
        if  ret is not OUTLIER:
            Z.append(X[i])
    #print('after : ', len(Z))

    Z = np.float32(Z)

    if len(Z)<3:
        continue

    if len(Z)>8*3:                     #24~32
        cluster_count = 8
    elif  len(Z)<=8*3 and len(Z)>4*3:#13~24
        cluster_count = 4
    elif len(Z)<=4*3 and len(Z)>4*2:#9~12
        cluster_count = 3
    elif len(Z)<=4*2 and len(Z)>4: # 5~8
        cluster_count = 2
    else: # 3~4
        cluster_count = 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,cluster_count,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    color=['#1f77b4','#aec7e8','#ff7f0e','r',
           '#2ca02c','#98df8a','r','#ff9896',
           '#9467bd','#c5b0d5','#8c564b','#c49c94']
    #fig, ax = plt.subplots(1, 1)
    for i in range(cluster_count):
        A = Z[label.ravel()==i]
        #print(i,A)
        #print('xmin : ', min(A[:,0]), 'xmax : ', max(A[:,0]))
        #print('ymin : ', min(A[:,1]), 'ymax : ', max(A[:,1]))

        xmin = min(A[:,0])-16
        xmax = max(A[:,0])+16
        ymin = min(A[:,1])-16
        ymax = max(A[:,1])+16
        w=xmax-xmin
        h=ymax-ymin

        #ax.scatter(A[:,0],A[:,1],c=color[i])
        #rect = plt.Rectangle((xmin-16, ymin-16), w+32, h+32, linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)


        newObject=ET.Element('object')
        newName=ET.Element('name')
        newName.text='crowd'
        newBBox=ET.Element('bndbox')
        xminObj = ET.Element('xmin')
        xminObj.text=str(int(xmin))
        yminObj = ET.Element('ymin')
        yminObj.text = str(int(ymin ))
        xmaxObj = ET.Element('xmax')
        xmaxObj.text = str(int(xmax ))
        ymaxObj = ET.Element('ymax')
        ymaxObj.text = str(int(ymax ))
        newBBox.append(xminObj)
        newBBox.append(xmaxObj)
        newBBox.append(yminObj)
        newBBox.append(ymaxObj)
        newDifficult = ET.Element('difficult')
        newDifficult.text = '0'
        newObject.append(newName)
        newObject.append(newBBox)
        newObject.append(newDifficult)
        root.append(newObject)

    list = glob.glob('/home/lee/Documents/Dataset_to_VOC_converter-master/train2/*.xml')
    newAnnotationPath=annotationPath.replace('train2','train4')
    tree.write(newAnnotationPath)

    #plt.Rectangle((min(A[:,0]),min(A[:,1])), (max(A[:,0])-min(A[:,0])), (max(A[:,1])-min(A[:,1])))
    #ax.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    #plt.xlabel('Height'),plt.ylabel('Weight')
    #plt.show()
