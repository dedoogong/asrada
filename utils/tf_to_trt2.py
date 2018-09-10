import tensorflow as tf
import tensorrt as trt
from tensorrt.parsers import uffparser

import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
from random import randint # generate a random test case
from PIL import Image
from matplotlib.pyplot import imshow #to show test case
import time #import system tools
import os

import sys
import uff

OUTPUT_NAMES="MatMul"

tf_model='/home/lee/models/frozen_model.pb'
uff_model = uff.from_tensorflow_frozen_model(tf_model, [OUTPUT_NAMES])
'''

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
parser = uffparser.create_uff_parser()
parser.register_input("Placeholder", (3,240,320), 0)
parser.register_output("OUTPUT_NAMES")
engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 31)




checkpoint = tf.train.get_checkpoint_state('')

input_checkpoint = checkpoint.model_checkpoint_path

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True

# We start a session using a temporary fresh Graph
with tf.Session(graph=tf.Graph()):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    graphdef = tf.get_default_graph().as_graph_def()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, OUTPUT_NAMES)
    tf_model = tf.graph_util.remove_training_nodes(frozen_graph)
    uff_model = uff.from_tensorflow(tf_model, OUTPUT_NAMES)
'''