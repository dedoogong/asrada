import os
import cv2 as cv
import csv
import numpy as np

csv_dir_1 = '/media/lee/ETC_300_150GB/FaceDB/object-dataset/labels.csv'
csv_root_dir_1 = '/media/lee/ETC_300_150GB/FaceDB/object-dataset/'

csv_dir_2 = '/media/lee/ETC_300_150GB/FaceDB/object-detection-crowdai/labels.csv'
csv_root_dir_2 = '/media/lee/ETC_300_150GB/FaceDB/object-detection-crowdai/'

with open(csv_dir_1 , 'r') as f:  # f == pts file
    reader = csv.reader(f, dialect='excel', delimiter=' ')
    for row in reader:
        imageFullPath=csv_root_dir_1+row[0]
        img = cv.imread(imageFullPath)
        img_height = float(img.shape[0])
        img_width = float(img.shape[1])

        ori_x_min = float(row[1])
        ori_y_min = float(row[2])
        ori_x_max = float(row[3])
        ori_y_max = float(row[4])

        front_back = int(row[5])
        label = row[6]
        class_id = -1

        if label == 'pedestrian':
            class_id = 0
        elif label == 'car':
            class_id = 1
        elif label == 'biker':
            class_id = 2
        elif label == 'truck':
            class_id = 5
        elif label == 'trafficLight':
            class_id = 6

        normed_cx = (ori_x_max+ori_x_min)/(2*img_width)
        normed_cy = (ori_y_max+ori_y_min)/(2*img_height)
        normed_w  = (ori_x_max-ori_x_min)/img_width
        normed_h  = (ori_y_max-ori_y_min)/img_height

        data=str( class_id ) + ' ' + str(normed_cx ) + ' ' +str(normed_cy ) + ' ' + str(normed_w ) + ' ' + str(normed_h ) + '\n'

        f1=open(imageFullPath.replace('jpg','txt'),'a')
        f1.write(data)
        f1.close()

with open(csv_dir_2, 'r') as f:  # f == pts file
    reader = csv.reader(f, dialect='excel', delimiter=' ')
    for row in reader:
        imageFullPath = csv_root_dir_2 + row[0].split(',')[4]
        img = cv.imread(imageFullPath)
        img_height = float(img.shape[0])
        img_width = float(img.shape[1])

        ori_x_min = float(row[0].split(',')[0])
        ori_y_min = float(row[0].split(',')[1])
        ori_x_max = float(row[0].split(',')[2])
        ori_y_max = float(row[0].split(',')[3])

        label = row[0].split(',')[5]

        class_id = -1

        if label == 'Pedestrian':
            class_id = 0
        elif label == 'Car':
            class_id = 1
        elif label == 'Truck':
            class_id = 5

        normed_cx = (ori_x_max + ori_x_min) / (2 * img_width)
        normed_cy = (ori_y_max + ori_y_min) / (2 * img_height)
        normed_w = (ori_x_max - ori_x_min) / img_width
        normed_h = (ori_y_max - ori_y_min) / img_height

        data = str(class_id) + ' ' + str(normed_cx) + ' ' + str(normed_cy) + ' ' + str(normed_w) + ' ' + str(normed_h) + '\n'

        f1 = open(imageFullPath.replace('jpg', 'txt'), 'a')
        f1.write(data)
        f1.close()