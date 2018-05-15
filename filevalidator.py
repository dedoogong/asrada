import glob
import os
import shutil

def annotation_image_matching_checker():
    list = glob.glob('/home/lee/Documents/Annotation/tmp/*.xml')
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
def file_mover():
    annotation_list = glob.glob('/home/lee/Documents/Annotation/tmp/*.xml')
    for annotationPath in annotation_list :
        imagePath=annotationPath.replace('xml', 'jpg')
        imagePath = imagePath.replace('/tmp', '')
        targetPath=imagePath.replace('/Annotation','/Annotation/tmp')
        if (os.path.exists(imagePath)):
            shutil.move(imagePath,targetPath)
        else:
            print(imagePath)

def main():
    annotation_image_matching_checker()
    #file_mover()

if __name__ == "__main__":
    main()