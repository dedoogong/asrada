from __future__ import print_function, division

import glob
import os
import shutil

import xml.etree.ElementTree as ET

def box_size_checker():

    list = glob.glob('/home/lee/Downloads/imgaug-master/annotations3/*.xml')

    for annotationPath in list:
        try:
            tree = ET.parse(annotationPath)
            root = tree.getroot()
            objects = root.findall ('object')

            for i in range(len(objects)):
                bndboxObj = objects[i].find('bndbox')
                xmin=int(bndboxObj.find('xmin').text)
                ymin=int(bndboxObj.find('ymin').text)
                xmax=int(bndboxObj.find('xmax').text)
                ymax=int(bndboxObj.find('ymax').text)

                objName=objects[i].find('name').text
                if objName=='license_plate' and (ymax-ymin) < 30:
                    print(root.find('filename').text, (ymax - ymin))

                    bndboxObj.find('ymin').text = (ymin - 10).__str__()
                    bndboxObj.find('ymax').text = (ymax + 10).__str__()

                if (xmax-xmin) < 20 or (ymax-ymin) < 10 :
                    print(root.find('filename'), (xmax-xmin), (ymax - ymin))
                    root.remove(objects[i])

            tree.write(annotationPath)
        except:
            print(annotationPath)

def main():
    box_size_checker()

if __name__ == "__main__":
    main()