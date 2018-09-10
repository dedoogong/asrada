from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.platform import gfile
from ColorHandPose3DNetwork import ColorHandPose3DNetwork
from general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d, plot_hand_2d
import cv2
import time
if __name__ == '__main__':
    image_list = list()
    WEBCAM = 0
    CROPPED = 0
    CROPPED_SHRUNKEN = 1
    # network input
    #image_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    #hand_side_tf = tf.constant([[1.0, 0.0]])
    #evaluation = tf.placeholder_with_default(True, shape=())
    #image_crop_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    #keypoints_scoremap_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 21))
    # build network
    #net = ColorHandPose3DNetwork()
    ###image_crop_tf, keypoints_scoremap_tf= net.inference(image_tf, hand_side_tf, evaluation)
    #keypoints_scoremap_tf = net.inference(image_tf, hand_side_tf, evaluation)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    #model_filename = '/home/lee/Downloads/Classify-HandGesturePose-master/pose2d/frozen_model.pb'
    model_filename = '/home/lee/Downloads/Classify-HandGesturePose-master/pose2d/quantized_graph.pb'

    with tf.gfile.GFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        image_tf = graph.get_tensor_by_name('import/Placeholder:0')
        keypoints_scoremap_tf = graph.get_tensor_by_name('import/ResizeBilinear:0')
        checkpoint_path = tf.train.latest_checkpoint("./pose2d/")
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #net.init(sess)
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
        saver.restore(sess, checkpoint_path)
        image_list.append('./data/test3.png')
        image_list.append('./data/test2.png')
        image_list.append('./data/test.png')

        for img_name in image_list:
            image_raw = scipy.misc.imread(img_name)
            t1 = time.clock()
            image_raw = scipy.misc.imresize(image_raw, (256,256))
            image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
            keypoints_scoremap_v = sess.run(['ResizeBilinear:0'], feed_dict={'Placeholder:0': image_v})
            image_crop_v = np.squeeze(image_v)
            keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
            image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
            coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
            print(time.clock() - t1)
            fig = plt.figure(1)
            ax2 = fig.add_subplot(111)
            plot_hand_2d(coord_hw_crop, image_crop_v)
            cv2.imshow('frame', image_crop_v)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        #saver = tf.train.Saver()
        #saver.save(sess, "./posemodel2d.ckpt" ,write_meta_graph=True )
        # Create a summary writer, add the 'graph' to the event file.
        # writer = tf.summary.FileWriter( './pose2d', sess.graph)
        # writer.add_graph(sess.graph)
        # Feed image list through network