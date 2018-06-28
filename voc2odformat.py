import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict
def generateVOC2Json(rootDir):
    attrDict = OrderedDict()
    images = list()

    for root, dirs, files in os.walk(rootDir):
        image_id = 0

        for file in files:
            image_id = image_id + 1
            annotation_path = os.path.abspath(os.path.join(root, file))
            doc = xmltodict.parse(open(annotation_path).read())
            annotations = list()
            if 'object' in doc['annotation']:
                for obj in doc['annotation']['object']:
                    annotation = OrderedDict()
                    x1 = int(obj["bndbox"]["xmin"])
                    y1 = int(obj["bndbox"]["ymin"])
                    w = int(obj["bndbox"]["xmax"]) - x1
                    h = int(obj["bndbox"]["ymax"]) - y1
                    annotation["box"] = [x1, y1, w, h]
                    annotation["occ"] = 0
                    annotation["tag"] = obj["name"]
                    ignore = OrderedDict()
                    ignore["ignore"] = 0
                    annotation["extra"] = ignore
                    annotations.append(annotation)
            attrDict["gtboxes"]=annotations
            imgpath=annotation_path.replace('.xml', '.jpg')
            attrDict["fpath"] =imgpath
            attrDict["dbName"] = "COCO"
            dbInfo = OrderedDict()
            dbInfo["vID"] = "0"
            dbInfo["frameID"] = "-1"
            attrDict["dbInfo"] = dbInfo
            attrDict["width"] = int(doc['annotation']['size']['width'])
            attrDict["height"] = int(doc['annotation']['size']['height'])
            attrDict["ID"] = imgpath.replace(rootDir+'/','')

            jsonString = json.dumps(attrDict)
            with open("pascal_parking_val.json", "a") as f:
            #with open("valid_parking.json", "a") as f:
                f.write(jsonString)
                f.write('\n')

rootDir = "/home/lee/Downloads/py-R-FCN/data/VOCdevkit0712/VOC0712/Annotations/val"
#rootDir = "/home/lee/Downloads/py-R-FCN/test/new_parking"
generateVOC2Json(rootDir )