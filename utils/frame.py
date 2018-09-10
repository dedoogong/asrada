import cv2
import os
import sys


path ='/home/lee/darknet/vid/'

for root, dirs, files in os.walk(path):

    for file in files:
        p=os.path.join(root,file)
        cap = cv2.VideoCapture(p)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("totalFrames : ", totalFrames)
        count = 0
        while True:
            count += 1
            myFrameNumber = 30 * count
            if myFrameNumber >= 0 and myFrameNumber <= int(totalFrames):
                # set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
                ret, frame = cap.read()
                if ret == True:
                    #print("myFrameNumber : ", myFrameNumber)
                    img_name = p[:-4] + '_' + str(count) + '.jpg'
                    cv2.imwrite(img_name, frame)
                else:
                    break
            else:
                break