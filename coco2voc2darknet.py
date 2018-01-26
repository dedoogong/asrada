import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob


sets=[('2014', 'train'), ('2014', 'val'), ('2017', 'train'), ('2017', 'val')]

classes = ["person","car","motorcycle","bus","train","truck","traffic light","stop sign","cat","dog","horse","parking meter","sheep","cow","bear","giraffe","cat2","zebra","cell phone"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(f):
    in_file = open(f)
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
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()



for year, image_set in sets:
    f=open('/media/lee/ImageDB_1000_100GB/ImageDB_coco/annotations/list.txt','r')
    lines = f.readlines()
    for l in lines:
        file = l.strip('\n')
        convert_annotation(file)

