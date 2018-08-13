# encoding: utf-8
"""
@author: Seunghyun Lee
@contact: dedoogong@gmail.com
"""
from __future__ import print_function, division
caffe_root = '/home/lee/Downloads/py-R-FCN/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

# load the data file
data_file = np.load('light_head_rfcn.npy')

# get the weights and biases out of the array
# the weights have to be transposed because of differences between Caffe and Tensorflow
# format filter weights:
# Tensorflow: [height (0), width (1), depth (2), number of filters (3)]
# Caffe:      [number of filters (3), depth (2), height (0), width (1)]
# define architecture
net = caffe.Net('vgg_net_19.prototxt', caffe.TEST)
# load parameters
for i in range(16):
    net.params['conv'+str(i+1)][0].data[...] = data_file[i][0].transpose((3,2,0,1))
    net.params['conv'+str(i+1)][1].data[...] = data_file[i][1]
# connecting the tensor after last pooling layer with the first fully-connected layer
# here is the link to the video where this part is explained (https://youtu.be/kvXHOIn3-8s?t=3m38s)
fc1_w = data_file[16][0].reshape((4, 4, 512, 4096))
fc1_w = fc1_w.transpose((3, 2, 0, 1))
fc1_w = fc1_w.reshape((4096, 8192))
fc1_b = data_file[16][1]
# Tensorflow: [number of inputs (0), number of outputs (1)]
# Caffe:      [number of outputs (1), number of inputs (0)]
fc2_w = data_file[17][0].transpose((1, 0))
fc2_b = data_file[17][1]

fc3_w = data_file[18][0].transpose((1, 0))
fc3_b = data_file[18][1]

net.params['fc1'][0].data[...] = fc1_w
net.params['fc1'][1].data[...] = fc1_b

net.params['fc2'][0].data[...] = fc2_w
net.params['fc2'][1].data[...] = fc2_b

net.params['fc3'][0].data[...] = fc3_w
net.params['fc3'][1].data[...] = fc3_b

# save caffemodel
net.save('light_head_rfcn.caffemodel')