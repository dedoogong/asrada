from __future__ import print_function, division
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
import glob
from collections import defaultdict
import PIL.Image
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(44)
ia.seed(44)

IMAGES_DIR = "myhand" #"myhand"

def main():
    list = glob.glob(r"D:\myhand/*.jpg")
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
            iaa.Fliplr(1.0), # horizontally flip 50% of all images
            iaa.Flipud(1.0), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.125, 0.75), "y": (0.125, 0.75)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
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
                    sometimes(iaa.PerspectiveTransform(scale=(0.075, 0.125)))
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
    image = ndimage.imread(img_path)
    textPath = img_path.replace('.jpg','.txt')
    f=open(textPath , 'r')
    rows_keypoints=[]
    classId=[]
    while True:
        line = f.readline()
        if not line: break
        values = line.split(' ')
        classId.append(values[0])
        cx= image.shape[1]*float(values[1])
        cy= image.shape[0]*float(values[2])
        w = image.shape[1]*float(values[3])
        h = image.shape[0]*float(values[4].replace('\n', ''))
        keypoints = [ia.Keypoint(x=(cx-w/2), y=(cy-h/2)), ia.Keypoint(x=(cx-w/2), y=(cy+h/2)), ia.Keypoint(x=(cx+w/2), y=(cy+h/2)), ia.Keypoint(x=(cx+w/2), y=(cy-h/2))]
        rows_keypoints.append(keypoints[0])
        rows_keypoints.append(keypoints[1])
        rows_keypoints.append(keypoints[2])
        rows_keypoints.append(keypoints[3])
    f.close()
    keypoints = [ia.KeypointsOnImage(rows_keypoints, shape=image.shape)]
    print("[draw_per_augmenter_images] Initializing...")
    rows_augmenters = [
        (0, "Crop\n(top, right,\nbottom, left)", [(str(vals), iaa.Crop(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        (0, "Pad\n(top, right,\nbottom, left)", [(str(vals), iaa.Pad(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        (0, "Fliplr", [(str(p), iaa.Fliplr(p)) for p in [1, 1, 1, 1, 1]]),
        (0, "Flipud", [(str(p), iaa.Flipud(p)) for p in [1, 1, 1, 1, 1]]),
        (0, "Add", [("value=%d" % (val,), iaa.Add(val)) for val in [-45, -25, 0, 25, 45]]),
        (0, "Add\n(per channel)", [("value=(%d, %d)" % (vals[0], vals[1],), iaa.Add(vals, per_channel=True)) for vals in [(-55, -35), (-35, -15), (-10, 10), (15, 35), (35, 55)]]),
        (0, "AddToHueAndSaturation", [("value=%d" % (val,), iaa.AddToHueAndSaturation(val)) for val in [-45, -25, 0, 25, 45]]),
        (0, "Multiply", [("value=%.2f" % (val,), iaa.Multiply(val)) for val in [0.25, 0.5, 1.0, 1.25, 1.5]]),
        (0, "GaussianBlur", [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma)) for sigma in [0.25, 0.50, 1.0, 2.0, 4.0]]),
        (0, "BilateralBlur\nsigma_color=250,\nsigma_space=250", [("d=%d" % (d,), iaa.BilateralBlur(d=d, sigma_color=250, sigma_space=250)) for d in [1, 3, 5, 7, 9]]),
        (0, "Sharpen\n(alpha=1)", [("lightness=%.2f" % (lightness,), iaa.Sharpen(alpha=1, lightness=lightness)) for lightness in [0, 0.5, 1.0, 1.5, 2.0]]),
        (0, "Emboss\n(alpha=1)", [("strength=%.2f" % (strength,), iaa.Emboss(alpha=1, strength=strength)) for strength in [0, 0.5, 1.0, 1.5, 2.0]]),
        (0, "AdditiveGaussianNoise", [("scale=%.2f*255" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        (0, "Dropout", [("p=%.2f" % (p,), iaa.Dropout(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "SaltAndPepper", [("p=%.2f" % (p,), iaa.SaltAndPepper(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "CoarseSaltAndPepper\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseSaltAndPepper(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "ContrastNormalization", [("alpha=%.1f" % (alpha,), iaa.ContrastNormalization(alpha=alpha)) for alpha in [0.5, 0.75, 1.0, 1.25, 1.50]]),
        (6, "PerspectiveTransform", [("scale=%.3f" % (scale,), iaa.PerspectiveTransform(scale=scale)) for scale in [0.075, 0.075, 0.10, 0.125, 0.125]]),
        (0, "Affine: Scale", [("%.1fx" % (scale,), iaa.Affine(scale=scale)) for scale in [0.1, 0.5, 1.0, 1.5, 1.9]]),
        (0, "Affine: Translate", [("x=%d y=%d" % (x, y), iaa.Affine(translate_px={"x": x, "y": y})) for x, y in [(int(-image.shape[1]*0.1), int( -image.shape[1]*0.1 )),
                                                                                                                     (int(-image.shape[1]*0.15),int( -image.shape[1]*0.1 )),
                                                                                                                     (int(-image.shape[1]*0.1), int( -image.shape[1]*0.15)),
                                                                                                                     (int( image.shape[1]*0.1), int(  image.shape[1]*0.1 )),
                                                                                                                     (int( image.shape[1]*0.15),int(  image.shape[1]*0.15))
                                                                                                                      ]]),
        (0, "Affine: Rotate", [("%d deg" % (rotate,), iaa.Affine(rotate=rotate)) for rotate in [-90, -75, -45, -30, -15, 0, 15, 30, 45, 75, 90]]),
        (0, "Affine: Shear", [("%d deg" % (shear,), iaa.Affine(shear=shear)) for shear in [-45, -25, 0, 25, 45]]),
        (0, "Affine: Modes", [(mode, iaa.Affine(translate_px=-32, mode=mode)) for mode in ["edge"]]),
        (
            2, "Affine: all", [
                (
                    "",
                    iaa.Affine(
                        scale={"x": (0.125, 0.75), "y": (0.125, 0.75)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-25, 25),  # shear by -16 to +16 degrees
                    )
                )
                for _ in sm.xrange(5)
            ]
        )
    ]
    print("[draw_per_augmenter_images] Augmenting...")
    rows = []
    for (row_seed, row_name, augmenters) in rows_augmenters:
        ia.seed(row_seed)
        row_images = []
        row_keypoints = []
        row_titles = []
        for img_title, augmenter in augmenters:
            aug_det = augmenter.to_deterministic()
            row_images.append(aug_det.augment_image(image))
            row_keypoints.append(aug_det.augment_keypoints(keypoints)[0])
            row_titles.append(img_title)
        rows.append((row_name, row_images, row_keypoints, row_titles))
    # routine to draw many single files
    seen = defaultdict(lambda: 0)
    markups = []
    m = 0
    for (row_name, row_images, row_keypoints, row_titles) in rows:
        #output_image = ExamplesImage(128, 128, 128+64, 32)
        row_images_kps = []
        for image, keypoints in zip(row_images, row_keypoints):
            #row_images_kps.append(keypoints.draw_on_image(image, size=15))
            m +=1
            print("[draw_per_augmenter_images] Saving augmented images...")
            misc.imsave("image_%05d_%05d.jpg" % (m, idx), image)
            ff = open("image_%05d_%05d.txt" % (m, idx), 'a')
            x = 0
            while True:
                if 4*x == len(keypoints.keypoints): break
                cx_ = (keypoints.keypoints[4*x].x + keypoints.keypoints[2+4*x].x) / (2 * image.shape[1])
                cy_ = (keypoints.keypoints[4*x].y + keypoints.keypoints[2+4*x].y) / (2 * image.shape[0])
                w_ = (keypoints.keypoints[2+4*x].x - keypoints.keypoints[4*x].x) / image.shape[1]
                h_ = (keypoints.keypoints[2+4*x].y - keypoints.keypoints[4*x].y) / image.shape[0]
                data = classId[x] + ' ' + str(cx_) + ' ' + str(cx_) + ' ' + str(cx_) + ' ' + str(cx_) + '\n'
                ff.write(data)
                x += 1
            ff.close()

class ExamplesImage(object):
    def __init__(self, image_height, image_width, title_cell_width, subtitle_height):
        self.rows = []
        self.image_height = image_height
        self.image_width = image_width
        self.title_cell_width = title_cell_width
        self.cell_height = image_height + subtitle_height
        self.cell_width = image_width

    def add_row(self, title, images, subtitles):
        assert len(images) == len(subtitles)
        images_rs = []
        for image in images:
            images_rs.append(ia.imresize_single_image(image, (self.image_height, self.image_width)))
        self.rows.append((title, images_rs, subtitles))

    def draw(self):
        rows_drawn = [self.draw_row(title, images, subtitles) for title, images, subtitles in self.rows]
        grid = np.vstack(rows_drawn)
        return grid

    def draw_row(self, title, images, subtitles):
        title_cell = np.zeros((self.cell_height, self.title_cell_width, 3), dtype=np.uint8) + 255
        title_cell = ia.draw_text(title_cell, x=2, y=12, text=title, color=[0, 0, 0], size=16)

        image_cells = []
        for image, subtitle in zip(images, subtitles):
            image_cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8) + 255
            image_cell[0:image.shape[0], 0:image.shape[1], :] = image
            image_cell = ia.draw_text(image_cell, x=2, y=image.shape[0]+2, text=subtitle, color=[0, 0, 0], size=11)
            image_cells.append(image_cell)

        row = np.hstack([title_cell] + image_cells)
        return row

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
