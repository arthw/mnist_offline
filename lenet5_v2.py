import os

op = 0
if op == 1:
  os.environ['MKL_NUM_THREADS']='1'
  # os.environ['MKLDNN_VERBOSE']=1
  # os.environ['MKL_VERBOSE']=1


import numpy as np

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras import backend as K

import keras

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import tensorflow as tf

import pre_show

tf.logging.set_verbosity(tf.logging.ERROR)

para_op = 0
if para_op == 1:
  config  = tf.ConfigProto ( inter_op_parallelism_threads=8,  intra_op_parallelism_threads=1 )

# config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
#                         allow_soft_placement=True, device_count = {'CPU': 1})
  session = tf.Session(config=config)
  K.set_session(session)



model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(32, 32, 1), \
    activation='tanh')) #C1
model.add(MaxPooling2D(pool_size=(2, 2)))#S2
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh'))#C3
model.add(MaxPooling2D(pool_size=(2, 2)))#S4
model.add(Flatten())
model.add(Dense(120, activation='tanh'))#C5
model.add(Dense(84, activation='tanh'))#F6
model.add(Dense(10, activation='softmax'))#output
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

