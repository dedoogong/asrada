import cv2

objectDatasetDir=r"C:\\Users\\dedoo\\Downloads\\object-detection-crowdai\\"
with open('labels.csv', 'r') as reader:
    for line in reader:
        print(line)
        items= line.strip().split(',')
        image=items[4]
        img  = cv2.imread(objectDatasetDir+image)
        imgH = img.shape[0]
        imgW = img.shape[1]
        xmin = float(items[0])
        xmax= float(items[1])
        ymin = float(items[2])
        ymax = float(items[3])
        label = items[5]
        classId=-1
        if label=='Car':
            classId = 1
        elif label == 'Truck':
            classId = 5
        elif label == 'Pedestrian':
            classId =0

        cx = float(xmin + xmax) / 2.0 / float(imgW)
        cy = float(ymin + ymax) / 2.0 / float(imgH)
        w = float(xmax - xmin) / float(imgW)
        h = float(ymax - ymin) / float(imgH)

        if classId >= 0:
            fp = open(objectDatasetDir + image.replace('.jpg', '.txt'), 'a')
            data=str(classId)+' ' + str(cx)+' '+str(cy)+' '+str(w)+' '+str(h)+ '\n'
            fp.write(data)
            fp.close()