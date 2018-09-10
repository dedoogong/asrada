from __future__ import print_function, unicode_literals
import pyyolo
import numpy as np
import sys
import cv2
import time
from FaceAlignment import FaceAlignment
import utils
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ColorHandPose3DNetwork import ColorHandPose3DNetwork
from general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d, plot_hand_2d

darknet_path = '/home/nvidia/Downloads/pyyolo/darknet'
datacfg = 'cfg/all.data'
#cfgfile = 'cfg/tiny-yolo-all.cfg'  ## cfg for real test (my custom 34 class)
cfgfile = 'cfg/tiny-yolo.cfg' ## cfg for webcam test (coco 80 class)
#cfgfile = 'cfg/yolo-all.cfg'
#weightfile = '/home/nvidia/Downloads/pyyolo/tiny-yolo_6200.weights'
#weightfile = '../yolo-all_12300.weights'
#weightfile = darknet_path + '/tiny-yolo_49000.weights'  ## cfg for real test (my custom 34 class)
weightfile = darknet_path + '/tiny-yolo.weights' ## cfg for webcam test (coco 80 class)  
thresh = 0.24
hier_thresh = 0.5

# GSTREAM PIPE for ON BOARD CAM

WEBCAM=0
if WEBCAM:
    gst_str = 'nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)416, height=(int)416, format=(string)I420, framerate=(fraction)120/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
    #flip 1  = -90 2 = -180 3=-270 4=lr 5=lr->90 6=lr->180 7=lr->280
    capture1= cv2.VideoCapture(gst_str)
    capture0 = cv2.VideoCapture(1)
    #capture1.set(cv2.CAP_PROP_FRAME_WIDTH,BackCamSize);
    #capture1.set(cv2.CAP_PROP_FRAME_HEIGHT,BackCamSize);

    capture0.set(cv2.CAP_PROP_FRAME_WIDTH,int(240))
    capture0.set(cv2.CAP_PROP_FRAME_HEIGHT,int(320))

else:
    capture1 = cv2.VideoCapture('face.mp4')
    capture0 = cv2.VideoCapture('hand.mp4')

BackCamSize=int(416/2)
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

#pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)
def init_FaceModel():
    model = FaceAlignment(112, 112, 1, 2)
    model.loadNetwork("DAN-Menpo.npz")
    return model

def init_HandModel():
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()

    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    return sess, \
           image_tf, hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
           keypoints_scoremap_tf, keypoint_coord3d_tf

def faceKeypointsDetector(faceImage,model):
    img = faceImage.astype(np.uint8)
    minX = 0
    maxX = faceImage.shape[1]
    minY = 0
    maxY = faceImage.shape[0]
    initLandmarks = utils.bestFitRect(None, model.initLandmarks, [minX, minY, maxX, maxY])

    if model.confidenceLayer:
        landmarks, confidence = model.processImg(img[np.newaxis], initLandmarks)
        if confidence < 0.1:
            reset = True
    else:
        landmarks = model.processImg(img[np.newaxis], initLandmarks)
    landmarks = landmarks.astype(np.int32)
    for i in range(landmarks.shape[0]):
        cv2.circle(faceImage, (landmarks[i, 0], landmarks[i, 1]), 2, (0, 255, 0))
    cv2.imshow("faceKeypoints",faceImage)
    cv2.waitKey(1)

def handKeypointsDetector(handImage, sess,
                          image_tf, hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                          keypoints_scoremap_tf, keypoint_coord3d_tf):

    image_raw = handImage
    newH = image_raw.shape[0]
    newW = image_raw.shape[1]
    image_raw = scipy.misc.imresize(image_raw, (newH, newW))
    blank_image = np.zeros((240, 320, 3), np.uint8)  # H,W,3
    offsetH = (240 - newH) / 2
    offsetW = (320 - newW) / 2
    blank_image[offsetH:(offsetH + image_raw.shape[0]), offsetW:(offsetW + image_raw.shape[1])] = image_raw
    image_v = np.expand_dims((blank_image.astype('float') / 255.0) - 0.5, 0)
    _, image_crop_v, _, _, \
    keypoints_scoremap_v, _ = sess.run(
        [hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
         keypoints_scoremap_tf, keypoint_coord3d_tf],
        feed_dict={image_tf: image_v})
    #hand_scoremap_v = np.squeeze(hand_scoremap_v)
    image_crop_v = np.squeeze(image_crop_v)                # ONLY image_crop_v
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)# AND  keypoints_scoremap_v NEEDED
    #keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    # post processing
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    #coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
    fig = plt.figure(1)
    fig.add_subplot(111)
    plot_hand_2d(coord_hw_crop, image_crop_v)
    #cv2.imshow('handKeypoints', image_crop_v)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

face_model=init_FaceModel()
image_tf, sess, hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
keypoints_scoremap_tf, keypoint_coord3d_tf=init_HandModel()

while (1):
    t1=time.clock()
    ret0,  frameBackOrig= capture0.read()
    #print('frameBackOrig : ',frameBackOrig.shape) # frameBackOrig :  (240, 320, 3)
    #print("web cam read time : ", t2-t1)
    
    t2=time.clock()
    ret1, frameFront= capture1.read()
    if not WEBCAM:
        frameFront=cv2.resize(frameFront, (416,416))
    #print('frameFront : ',frameFront.shape) # frameFront :  (416, 416, 3)
    #print("on-board cam read time : ", time.clock()-t2) 
    
    ####### Mapping Backward Webcam Frame to Forward Frame's Top, Left ##########
    # 1. Crop & Resize frame 
    frameBack  = frameBackOrig[ int(frameBackOrig.shape[0]*0.1) : int(frameBackOrig.shape[0]*0.9), int(frameBackOrig.shape[1]*0.2):frameBackOrig.shape[1]] 
    frameBack_W=int(BackCamSize)
    frameBack_H=int(BackCamSize) 
    frameBack  = cv2.resize(frameBack, (208, 208)) 
    frameBack=cv2.flip(frameBack,1)
    #print('frameBack : ',frameBack.shape) # frameBack :  (208, 208, 3)
    # 2. Map B-frame to F-frame
    
    ####### Extract ROI from Forward Frame's Center and Mapping it to Forward Frame's Top, Right ##########
    # 1. Crop & Resize frame 
    BackCamSize=416
    t = int(BackCamSize/2 - BackCamSize/32)
    l = int(BackCamSize/2 - BackCamSize/4)
    b = int(BackCamSize/2 + BackCamSize/8 + BackCamSize/16)
    r = int(BackCamSize/2 + BackCamSize/4)

    #cv2.rectangle(frameFront,(l, t),(r,b),255,5)
    frameROI = frameFront[t:b, l:r]
    frameROI_W=int(208)
    frameROI_H=int(208)
    #frameROI = cv2.resize(frameROI, (frameROI_W, frameROI_H), interpolation=cv2.INTER_LANCZOS4)
    frameROI = cv2.resize(frameROI, (frameROI_W, frameROI_H))
    
    frameFront[0 : 208, 0 : 208] = frameBack    
    frameFront[0 : 208, 208: 416] = frameROI
    img = frameFront.transpose(2,0,1)
    c, h, w = img.shape[0], img.shape[1], img.shape[2] 
    data = img.ravel()/255.0
    data = np.ascontiguousarray(data, dtype=np.float32)
    #output = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
    #boundingBoxes = []
    #for outputs in output:
    #    boundingBoxes.append([int(outputs['left']), int(outputs['top']), int(outputs['right']), int(outputs['bottom'])])
    #pick = np.asarray(boundingBoxes)
    #pick = non_max_suppression_fast(pick, 0.5)
    #for p in pick:
    #    cv2.rectangle(frameFront, (p[0],p[1]),(p[2],p[3]),(0,255,0), 3)
    #print(pick) 
    
	# print ("[x] before NMS, %d bounding boxes" % (len(pick )))
    #faceImage=[]
    #handImage=[]
    #faceKeypointsDetector(faceImage, face_model)
    #handKeypointsDetector(handImage, sess, image_tf, hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf)
    #cv2.imshow('webcam', frameFront)
    print("Vi -> Darknet -> Vo total FPS : ", 1/(time.clock()-t1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

capture0.release() 
capture1.release()
cv2.destroyAllWindows() 
pyyolo.cleanup()
