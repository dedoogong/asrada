from __future__ import print_function, division

import xml.etree.ElementTree as ET
import pickle

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import gridspec
import six
import six.moves as sm
import re
import os
from os import listdir, getcwd
from os.path import join
import cv2
import glob
from collections import defaultdict
import PIL.Image
import itertools

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(44)
ia.seed(44)

IMAGES_DIR = "myhand" #"myhand"

def main():
    list = glob.glob('/home/lee/Documents/carDB/2/*.jpg')
    for i in range(len(list)):
        #draw_single_sequential_images(list[i],i)
        draw_per_augmenter_images(list[i],i)

def draw_single_sequential_images(img_path,idx):
    ia.seed(44)
    image = ndimage.imread(img_path)
    #image = ia.quokka_square(size=(128, 128))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.125, 0.75), "y": (0.125, 0.75)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}, # translate by -20 to +20 percent (per axis)
                rotate=(-30, 30), # rotate by -45 to +45 degrees
                shear=(-25, 25), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((1, 2.0)), # blur images with a sigma between 0 and 3.0
                    ]),
                    iaa.Sharpen(alpha=(1.0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.05, 0.1*255), per_channel=0.5), # add gaussian noise to images
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    #grid = seq.draw_grid(image, cols=8, rows=8)
    #misc.imsave("examples_grid.jpg", grid)

    images = [image] * 10
    #TODO : convert xml to txt and read it in yolo format and replace the x, y coordinates with them.
    keypoints = [ia.Keypoint(x=34, y=15), ia.Keypoint(x=85, y=13), ia.Keypoint(x=63, y=73)]  # left ear, right ear, mouth

    keypointsList=[keypoints] * 10
    aug_det = seq.to_deterministic()
    images_aug = aug_det.augment_images(images)

    row_keypoints = []
    image_keypoints = []
    for idx in range(len(keypointsList)):
        onImageKeypoints = [ia.KeypointsOnImage(keypointsList[idx], shape = image.shape)]
        keypoints_aug = aug_det.augment_keypoints(onImageKeypoints)
        row_keypoints.append(keypoints_aug[0])

    for image, keypoints in zip(images_aug, row_keypoints):
        image_keypoints.append(keypoints.draw_on_image(image, size=5))

    for i, image_aug in enumerate(image_keypoints):
        misc.imsave("image_%05d_%06d.jpg" % (idx,i), image_aug)

def draw_per_augmenter_images(img_path,idx):
    print("[draw_per_augmenter_images] Loading image...")
    #res=ndimage.imread(img_path)
    #image = np.reshape(res,[1,res.shape[0],res.shape[1],3])
    image = ndimage.imread(img_path)
    image2=image
    print(img_path)
    xmlPath = img_path.replace('.jpg','.xml')

    tree = ET.parse(xmlPath)
    root = tree.getroot()
    size = root.find('size')
    filename=root.find('filename').text
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    objects = root.findall('object')
    keypoints = []
    onImageKeypoints = []

    for i in range(len(objects)):
        bndboxObj = objects[i].find('bndbox')
        xmin=int(bndboxObj.find('xmin').text)
        ymin=int(bndboxObj.find('ymin').text)
        xmax=int(bndboxObj.find('xmax').text)
        ymax=int(bndboxObj.find('ymax').text)
        keypoints.append([ia.Keypoint(x=xmin,y=ymin), ia.Keypoint(x=xmin,y=ymax), ia.Keypoint(x=xmax,y=ymin), ia.Keypoint(x=xmax,y=ymax)])
    keypoints = list(itertools.chain.from_iterable(keypoints))
    onImageKeypoints.append(ia.KeypointsOnImage(keypoints, shape=image.shape))
    print("[draw_per_augmenter_images] Initializing...")
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(0.1, 0.2),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.5, 1.0), "y": (0.5, 1.0)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15),  # rotate by -45 to +45 degrees
            order=[0],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode='edge'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.Emboss(alpha=(1.0), strength=(0.5, 1.0)),  # emboss images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.03, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.2, 0.3), per_channel=0.2),
                       ]),
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               second=iaa.ContrastNormalization((0.5, 2.0))
                           )
                       ]),
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.04))),  # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
        random_order=True
    )

    for aug_count in range(100):
        print("Augmenting...")
        seq_det = seq.to_deterministic()
        # augment keypoints and images
        images_aug = seq_det.augment_images([image])
        keypoints_aug = seq_det.augment_keypoints(onImageKeypoints)
        print("Augmented...")
        m = 0
        for image_aug, keypoint_aug in zip(images_aug, keypoints_aug):
            m += 1
            boxCount = 0
            for i in range(len(objects)):
                bndboxObj = objects[i].find('bndbox')
                newXmin = min(int(keypoint_aug.keypoints[4 * boxCount].x), int(keypoint_aug.keypoints[3 + 4 * boxCount].x))
                newYmin = min(int(keypoint_aug.keypoints[4 * boxCount].y), int(keypoint_aug.keypoints[3 + 4 * boxCount].y))
                newXmax = max(int(keypoint_aug.keypoints[4 * boxCount].x), int(keypoint_aug.keypoints[3 + 4 * boxCount].x))
                newYmax = max(int(keypoint_aug.keypoints[4 * boxCount].y), int(keypoint_aug.keypoints[3 + 4 * boxCount].y))

                bndboxObj.find('xmin').text = newXmin.__str__()
                bndboxObj.find('xmax').text = newXmax.__str__()
                bndboxObj.find('ymin').text = newYmin.__str__()
                bndboxObj.find('ymax').text = newYmax.__str__()

                #try:
                #    cv2.rectangle(image, (int(newXmin ), int(newYmin )), (int(newXmax ), int(newYmax )), (0, 255, 0), 25)
                #except:
                #    image = image.transpose((1, 2, 0)).astype(np.uint8).copy()
                #    cv2.rectangle(image, (int(newXmin), int(newYmin)), (int(newXmax), int(newYmax)), (0, 255, 0), 25)
                #    image = image.transpose((2, 0, 1)).astype(np.uint8).copy()
                boxCount += 1
                #image=cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                #cv2.imshow('test2', image)
                #cv2.waitKey(1000)
                filename_=filename.replace('.jpg','')
            tree.write("annotations/%s_%02d_%02d_%02d.xml" % (filename_,aug_count,m,boxCount))
            misc.imsave("images/%s_%02d_%02d_%02d.jpg" % (filename_,aug_count,m, boxCount), image_aug)

def compress_to_jpg(image, quality=75):
    quality = quality if quality is not None else 75
    im = PIL.Image.fromarray(image)
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    jpg_string = out.getvalue()
    out.close()
    return jpg_string

def decompress_jpg(image_compressed):
    img_compressed_buffer = BytesIO()
    img_compressed_buffer.write(image_compressed)
    img = ndimage.imread(img_compressed_buffer, mode="RGB")
    img_compressed_buffer.close()
    return img

def arrdiff(arr1, arr2):
    nb_cells = np.prod(arr2.shape)
    d_avg = np.sum(np.power(np.abs(arr1.astype(np.float64) - arr2.astype(np.float64)), 2)) / nb_cells
    return d_avg

def save(fp, image, quality=75):
    image_jpg = compress_to_jpg(image, quality=quality)
    image_jpg_decompressed = decompress_jpg(image_jpg)

    # If the image file already exists and is (practically) identical,
    # then don't save it again to avoid polluting the repository with tons
    # of image updates.
    # Not that we have to compare here the results AFTER jpg compression
    # and then decompression. Otherwise we compare two images of which
    # image (1) has never been compressed while image (2) was compressed and
    # then decompressed.
    if os.path.isfile(fp):
        image_saved = ndimage.imread(fp, mode="RGB")
        #print("arrdiff", arrdiff(image_jpg_decompressed, image_saved))
        same_shape = (image_jpg_decompressed.shape == image_saved.shape)
        d_avg = arrdiff(image_jpg_decompressed, image_saved) if same_shape else -1
        if same_shape and d_avg <= 1.0:
            print("[INFO] Did not save image '%s', because the already saved image is basically identical (d_avg=%.8f)" % (fp, d_avg,))
            return
        else:
            print("[INFO] Saving image '%s'..." % (fp,))

    with open(fp, "w") as f:
        f.write(image_jpg)

if __name__ == "__main__":
    main()