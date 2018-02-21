# asrada

Download gits N follow their installation, respectively
1. DeepSpeech : https://github.com/mozilla/DeepSpeech
2. DeepHand : https://github.com/lmb-freiburg/hand3d
3. DeepAlignmentNetwork(Theano) : https://github.com/MarekKowalski/DeepAlignmentNetwork
    if you dont want to use DAN, you can use OpenFace instead. 
    https://github.com/TadasBaltrusaitis/OpenFace

4. Darknet : https://github.com/pjreddie/darknet
5. OpenPose : https://github.com/CMU-Perceptual-Computing-Lab/openpose

Download pretrained models and save each model to each project's model folder
and download detection model for darknet and all other files(names, cfg,data,avi)
> https://www.dropbox.com/sh/9r0lju9ju2nlof4/AACxeIxOOZMhrTc23p6RVXmOa?dl=0

move all.names to data folder, all.data and tiny-yolo.cfg to cfg folder in darknet folder.
 
detection test : ./darknet detector demo cfg/all.data cfg/tiny-yolo.cfg tiny-yolo_49000.backup output.avi -i 0

By optimizing darknet, I got 3 times faster results with tiny-yolo in all process.
It runs in 160~200 FPS(not only prediction step but including all pre/post image processing step).
Without Opt, normally tiny-yolo runs in 50~60 FPS. 
It was tested on i7 and GTX 1080.

Roughly, it runs in 50~60 FPS on Jetson TX2.

TODO : 
1. replace tiny-yolo's feature extraction with mobilenet based darknet. 
2. change FP32 to FP16
3. apply prunning

I guess after all opts are applied, it would run in more than 100 FPS on TX2 and model size would be less than 10MB with the same mAP(around 70~80). 
 
