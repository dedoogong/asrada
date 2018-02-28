# asrada

For testing car/hand/face/traffic sign detection and gesture control or free talking, you must connect one usb webcam and i2s mic/speaker like ReSpeaker/Breakout board to TX2.

Download asurada_detector.zip and darknet.zip, then unzip both and move darknet folder into asurada_detector folder. 

Run, r.sh, it will install all(I refered pyyolo)

$./r.sh

Download pretrained models and save each model to each project's model folder
and download detection model for darknet and all other files(names, cfg,data,avi)
> https://www.dropbox.com/sh/9r0lju9ju2nlof4/AACxeIxOOZMhrTc23p6RVXmOa?dl=0


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

TODO : More optimization for speed!

1. replace tiny-yolo's feature extraction with mobilenet based darknet. 
2. change FP32 to FP16
3. apply prunning
4. optimizing imread/resize node with nvx 
I guess after all opts are applied, it would run in more than 50 FPS on TX2 and model size would be less than 10MB with the same mAP(around 70~80). 
 
