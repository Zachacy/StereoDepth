#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import division

# only keep warnings and errors
import os,cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
from pydnet import *
import time
parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--dataset',           type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--left',        type=str,   help='path to the Left image', required=True)
parser.add_argument('--right',       type=str,   help='path to the Right image', required=True)
parser.add_argument('--output_directory',  type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='.')
parser.add_argument('--checkpoint_dir',        type=str,   help='path to a specific checkpoint to load', default='checkpoint/IROS18/pydnet')
parser.add_argument('--resolution',        type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')

args = parser.parse_args()

def read_image(image_path):
    image  = tf.image.decode_image(tf.read_file(image_path))
    image.set_shape( [None, None, 3])
    image  = tf.image.convert_image_dtype(image,  tf.float32)
    image  = tf.expand_dims(tf.image.resize_images(image,  [256, 512], tf.image.ResizeMethod.AREA), 0)

    return image

def test(params):

    img_l = read_image(args.left)
    img_r = read_image(args.right)
    img = tf.concat([img_l, img_r], 3)
    placeholders = {'im0':img}
    #placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 6], name='im0')}

    with tf.variable_scope("model") as scope:    
      model = pydnet(placeholders)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    train_saver.restore(sess, args.checkpoint_dir)

    # GET TEST IMAGES NAMES
    t0 = time.time()
    disp = sess.run(model.results[args.resolution-1])
    print('%.2f'%(time.time()-t0))
    disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
    #toShow = (np.concatenate((img[0], disp_color), 0)*255.).astype(np.uint8)# Concatenation between RGB and Depth image
    #toShow = cv2.resize(toShow, (int(width/2), int(height)))

    #cv2.imshow('pydnet', disp_color)
    #cv2.imwrite("disparity.jpg",disp_color*255)
    #cv2.waitKey(0)
    #print(disp)
    '''
    disparities = np.zeros((samples, 256, 512), dtype=np.float32)
    for step in range(samples):
        print('Running %d out of %d'%(step, samples))

        # If you want to evaluate lower resolution results, just get results[1] or results[2]
        disp = sess.run(model.results[args.resolution-1])
        disparities[step] = disp[0,:,:,0].squeeze()
    '''

def main(_):

    test(args)

if __name__ == '__main__':
    tf.app.run()
