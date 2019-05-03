import glob
import os
import shutil

def annotation_image_matching_checker2():
    annotationsList = glob.glob('/home/lee/Downloads/py-R-FCN/data/VOCdevkit2007/VOC2007/Annotations/*.xml')
    for each_annotationPath in annotationsList :
        jpgPath=each_annotationPath.replace('xml', 'jpg')
        jpgPath=jpgPath.replace('Annotations', 'JPEGImages')
        if (os.path.exists(jpgPath)):
            continue
        else:
            print(jpgPath)

    jpgList = glob.glob('/home/lee/Downloads/py-R-FCN/data/VOCdevkit2007/VOC2007/JPEGImages/*.jpg')
    for each_jpgPath in jpgList :
        annotationPath=each_jpgPath.replace('jpg', 'xml')
        annotationPath=annotationPath.replace('JPEGImages', 'Annotations')
        if (os.path.exists(annotationPath)):
            continue
        else:
            print(annotationPath)

def annotation_image_matching_checker():
    list = glob.glob('/home/lee/Downloads/py-R-FCN-master/data/VOCdevkit0712/VOC0712/Annotations/*.xml')
    for annotationPath in list:
        if (os.path.exists(annotationPath.replace('xml', 'jpg'))):
            continue
        else:
            print(annotationPath)

    list = glob.glob('/home/lee/Documents/Annotation/tmp/*.jpg')
    for imagePath in list:
        if (os.path.exists(imagePath.replace('jpg', 'xml'))):
            continue
        else:
            print(imagePath)
def jpg_file_mover():
    annotation_list = glob.glob('/home/lee/Downloads/imgaug-master/annotations/*.xml')
    for annotationPath in annotation_list :
        imagePath=annotationPath.replace('xml', 'jpg')
        imagePath = imagePath.replace('annotations', 'images')
        targetPath=imagePath.replace('images','images_checked')

        if (os.path.exists(imagePath)):
            shutil.move(imagePath,targetPath)
        else:
            print(imagePath)

def xml_file_mover():
    image_list = glob.glob('/home/lee/Downloads/imgaug-master/images2/*.jpg')
    for imagePath in image_list :
        annotationPath=imagePath.replace('jpg', 'xml')
        annotationPath = annotationPath.replace('images2', 'annotations')
        targetPath=annotationPath.replace('annotations','annotations2')

        if (os.path.exists(annotationPath)):
            shutil.move(annotationPath,targetPath)
        else:
            print(annotationPath)
def main():
    annotation_image_matching_checker2()
    #xml_file_mover()

if __name__ == "__main__":
    main()