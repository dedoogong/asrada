import cv2
import os
import sys

'''
#with open('labels.csv', 'r') as reader:
#    for line in reader:
#        print(line)
'''
objectDatasetDir=r"C:\\Users\\dedoo\\Downloads\\object-dataset\\"

with open('labels-object-dataset.csv', 'r') as reader:
    for line in reader:
        fields = line.strip().split(',')
        items=fields[0].split(' ')
        image=items[0]
        img  = cv2.imread(objectDatasetDir+image)
        imgH = img.shape[0]
        imgW = img.shape[1]
        xmin = float(items[1])
        ymin = float(items[2])
        xmax = float(items[3])
        ymax = float(items[4])
        label = items[6].split("\"")[1]
        classId=-1

        if label=='car':
            classId = 1
        elif label=='trafficLight':
            classId = 6
        elif label == 'truck':
            classId = 5
        elif label == 'pedestrain':
            classId =0
        elif label == 'biker':
            classId = 2

        #if(len(items)) == 8:
        #    lightColor = items[7]

        cx = float(xmin + xmax) / 2.0 / float(imgW)
        cy = float(ymin + ymax) / 2.0 / float(imgH)
        w = float(xmax - xmin) / float(imgW)
        h = float(ymax - ymin) / float(imgH)

        print(fields)

        if classId >= 0:
            fp = open(objectDatasetDir + image.replace('.jpg', '.txt'), 'a')
            data=str(classId)+' ' + str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+ '\n'
            fp.write(data)
            fp.close()