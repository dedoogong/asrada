# asrada

Download gits N follow their installation, respectively
1. DeepSpeech : https://github.com/mozilla/DeepSpeech
2. DeepHand : https://github.com/lmb-freiburg/hand3d
3. DeepAlignmentNetwork(Theano) : https://github.com/MarekKowalski/DeepAlignmentNetwork
    if you dont want to you DAN, you can use OpenFace instead. 
    https://github.com/TadasBaltrusaitis/OpenFace

4. Darknet : https://github.com/pjreddie/darknet
5. OpenPose : https://github.com/CMU-Perceptual-Computing-Lab/openpose

Download pretrained models and save each model to each project's model folder
and download detection model for darknet and all other files(names, cfg,data,avi)
> https://www.dropbox.com/sh/9r0lju9ju2nlof4/AACxeIxOOZMhrTc23p6RVXmOa?dl=0

move all.names to data folder, all.data and tiny-yolo.cfg to cfg folder in darknet folder.
 
detection test : ./darknet detector demo cfg/all.data cfg/tiny-yolo.cfg tiny-yolo_49000.backup output.avi -i 0
