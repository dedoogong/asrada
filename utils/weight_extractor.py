# encoding: utf-8
"""
@author: Seunghyun Lee
@contact: dedoogong@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython import embed
from config import cfg, config

import dataset
import os.path as osp
import network_desp
import tensorflow as tf
from tqdm import tqdm
from utils.py_utils import misc

if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    args.devices = misc.parse_devices(args.devices)
    if args.end_epoch == -1:
        args.end_epoch = args.start_epoch

    devs = args.devices.split(',')
    misc.ensure_dir(config.eval_dir)
    eval_file = open(os.path.join(config.eval_dir, 'results.txt'), 'a')
    dataset_dict = dataset.val_dataset()
    records = dataset_dict['records']
    nr_records = len(records)
    read_func = dataset_dict['read_func']

    nr_devs = len(devs)
    for epoch_num in range(args.start_epoch, args.end_epoch + 1):
        model_file = osp.join(
            config.output_dir, 'model_dump','epoch_{:d}'.format(epoch_num) + '.ckpt')
        pbar = tqdm(total=nr_records)
        all_results = []

        if nr_devs > 1:
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                os.environ["CUDA_VISIBLE_DEVICES"] = devs[0]
                tfconfig = tf.ConfigProto(allow_soft_placement=True)
                tfconfig.gpu_options.allow_growth = True
                sess = tf.Session(config=tfconfig)
                net = network_desp.Network()
                inputs = net.get_inputs()
                net.inference('TEST', inputs)
                saver = tf.train.Saver()
                saver.restore(sess, model_file)

                vars = tf.trainable_variables()
                for v in vars:
                    print('======================================================================')
                    print(v.op.name)
                    op = sess.graph.get_tensor_by_name(v.op.name+':0')
                    res=sess.run(op)
                    print(res)