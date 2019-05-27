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


class ShowPerf(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        pass
        # print(logs)
        
showperf=ShowPerf()

classes = 10

# X_train = create_x(55000, './train/')
# X_test = create_x(10000, './test/')

# Y_train = create_y(classes, 'train.txt')
# Y_test = create_y(classes, 'test.txt')

mnist = input_data.read_data_sets("./")
X_train, Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = np.array(Y_train).astype('float32')
Y_test = np.array(Y_test).astype('float32')

Y_train = np_utils.to_categorical(Y_train, classes)
Y_test = np_utils.to_categorical(Y_test, classes)

print(X_train.shape, X_test.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#print(X_train[0])



model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), \
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
history = model.fit(X_train, Y_train, batch_size=55000, epochs=3, verbose=1, \
    validation_data=(X_test, Y_test), callbacks=[showperf])
score = model.evaluate(X_test, Y_test, verbose=0)

test_result = model.predict(X_test)
result = np.argmax(test_result, axis=1)

#print(result)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#print the model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)