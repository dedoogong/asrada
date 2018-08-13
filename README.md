# asrada
-----------------------------------------------------------------------------------------------------------------------------

light head rcnn TF to caffe converting (on testing)

**tf2caffe.py
weight_extractor.py**

-----------------------------------------------------------------------------------------------------------------------------
For testing car/hand/face/traffic sign detection and gesture control or free talking, you must connect one usb webcam and i2s mic/speaker like ReSpeaker/Breakout board to TX2.

Download asurada_detector.zip and darknet.zip, then unzip both and move darknet folder into asurada_detector folder. 

Run, r.sh, it will install all(I refered pyyolo)

$./r.sh

Download pretrained models and save each model to each project's model folder
and download detection model for darknet and all other files(names, cfg,data,avi)
> https://www.dropbox.com/sh/9r0lju9ju2nlof4/AACxeIxOOZMhrTc23p6RVXmOa?dl=0


## UPDATE
I succeed in converting yolov2 model to caffe version. I uploaded it on naver cloud too. I'm trying to  convert it to tensorRT! (I think  caffe based TRT is faster than TF based TRT). I'm trying to get the detection box based on python again!
yolov2 caffemodel : http://naver.me/GWGiBG8R
yolov2 prototxt : http://naver.me/5JoGu38j

![GitHub Logo](/NMS/person_results.png) 
![GitHub Logo](/NMS/horses_results.png) 
![GitHub Logo](/NMS/eagle_results.png) 
![GitHub Logo](/NMS/giraffe_results.png) 
![GitHub Logo](/NMS/dog_results.png) 





I  recommand to use python3, not python2. 

$python3 ./test_2webcam.py

Roughly, it runs in 25 FPS(depending on how many objs are found because of get_region_box and NMS) on Jetson TX2.

For others, download gits N follow their installation, respectively
1. DeepSpeech : https://github.com/mozilla/DeepSpeech
2. DeepHand : https://github.com/lmb-freiburg/hand3d
3. DeepAlignmentNetwork(Theano) : https://github.com/MarekKowalski/DeepAlignmentNetwork
    if you dont want to use DAN, you can use OpenFace instead. 
    https://github.com/TadasBaltrusaitis/OpenFace
4. OpenPose : https://github.com/CMU-Perceptual-Computing-Lab/openpose

In case of DAN, I replaced the existing Theano based DAN with TF based one for optimization. Please refer to DataPrepare.py for DB setup in npz format. run DAN.py for training. I uploaded pre built image set in npz format in dropbox. 

## UPDATE
It runs > 25 FPS. I modified the existing net architecture a lot. I removed all stage 2 net and replaced all conv with separable dw conv(270MB->7MB!!). it runs x2 faster!!
Please use DAN_stage1_spdw.py instead of DAN.py
 
In case of DeepHand, please download quantized_graph.pb for testing. 
 I use just middle part of the whole model, 'PoseNet'. I removed 'HandSegNet' and 'PosePriorNet, ViewPointNet' and I quantized the 'PoseNet' part to get the reduced and faster model. Size is changed from 188.4MB(2 pickles) -> 70 MB(1 frozen pb) -> 17.6 MB(1 quantized_graph.pb)

In case of DeepSpeech, you need to modify some sourced to build it on TX2. Please read really carefully of the instruction. 
I had such a hard time to build it finally successfully. 
Bcus of the RNN(bidirection network connection), it fails in quantization in TF or TRT. But fortunately, I can run it using GPUs! (Original version just support only armv6 CPU).

TODO : More optimization for speed!

1. replace tiny-yolo's feature extraction with mobilenet based darknet. 
2. change FP32 to FP16
3. apply prunning
4. optimizing imread/resize node with nvx 
I guess after all opts are applied, it would run in more than 50 FPS on TX2 and model size would be less than 10MB with the same mAP(around 70~80). 
 
