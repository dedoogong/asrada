#!/usr/bin/env bash
python anno_coco2voc.py --anno_file ./annotations/instances_train2014.json \
                        --type instance \
                        --output_dir ./instance_train_annotation
python anno_coco2voc.py --anno_file ./annotations/instances_val2014.json \
                        --type instance \
                        --output_dir ./instance_val_annotation

python anno_coco2voc.py --anno_file ./annotations/instances_train2017.json \
                        --type instance \
                        --output_dir ./instance_train_annotation2
python anno_coco2voc.py --anno_file ./annotations/instances_val2017.json \
                        --type instance \
                        --output_dir ./instance_val_annotation2
#python anno_coco2voc.py --anno_file /startdt_data/COCO/dataset/annotations/person_keypoints_train2014.json \
#                        --type keypoint \
#                        --output_dir /startdt_data/COCO/dataset/keypoints_train_annotation
#python anno_coco2voc.py --anno_file /startdt_data/COCO/dataset/annotations/person_keypoints_val2014.json \
#                        --type keypoint \
#                        --output_dir /startdt_data/COCO/dataset/keypoints_val_annotation