import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
#classes = ["person","car","motorcycle","bus","train","truck","traffic light","stop sign","cat","dog","horse","parking meter","sheep","cow","bear","giraffe","cat2","zebra","cell phone"]
classes = ["brakeLight","trafficRedLight","brakeLightOFF","trafficGreenLight","trafficYellowLight","car","bus","truck","motorbike","trafficTurnLeft",
           "carFront","van","farBrakeCar","pedestrian","face","hand"]
"""
0   2 
 5 6 7 
8     11 
  14 15
"""
"brakeLight", "trafficRedLight", "brakeLightOFF", "trafficGreenLight", \
"trafficYellowLight", "car", "bus", "truck", \
"motorbike", "trafficTurnLeft", "carFront", "van", \
"farBrakeCar", "pedestrian", "face", "hand" \

"smallbrakeLight", "midbrakeLight", \
"smallbrakeLightOFF", "midbrakeLightOFF", \
"smallCar", "midCar", \
"smallbus", "midbus", \
"smalltruck", "midtruck", \
"smallmotorbike", "midmotorbike" \
"smallVan", "midVan" \
"smallFace", "midFace", \
"smallHand", "midHand"

    # 24classes
def convert(size, box):

    dw = 1./1280
    dh = 1./720
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def smallOrMid(w,h):
    if (w < 0.5 / (13.0) or h < 0.5 / (13.0)):  # small
        return "too_small"
    elif (w < 1 / (13.0) or h < 1 / (13.0)):  # small
        return "small"
    elif ((w < 3.0 / (13.0) and w > 1 / (13.0)) or (h < 3.0 / (13.0) and h > 3.0 / (13.0))):  # mid
        return "middle"

def convert_annotation(f):
    in_file = open(f)
    if os.path.exists(f.replace('xml','txt')):
        return
    out_file = open(f.replace('xml','txt'), 'a')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)

        if (cls_id == 0):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 16
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 17
        elif (cls_id == 2):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 18
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 19
        elif (cls_id == 5):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 20
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 21
        elif (cls_id == 6):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 22
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 23
        elif (cls_id == 7 ):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 24
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 25
        elif (cls_id == 8):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 26
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 27
        elif (cls_id == 11):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 28
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 29
        elif (cls_id == 14):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 30
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 31
        elif (cls_id == 15):
            if smallOrMid(bb[2], bb[3]) == "small":
                cls_id = 32
            elif smallOrMid(bb[2], bb[3]) == "middle":
                cls_id = 33

        if smallOrMid(bb[2], bb[3]) == "too_small":
            continue

        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()
#f=open('D:/myhand/myhand.txt','r')
f=open('D:/mycar/anno/list.txt','r')
lines = f.readlines()
for l in lines:
    file = l.strip('\n')
    convert_annotation(file)