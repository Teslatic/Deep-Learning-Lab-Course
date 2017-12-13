#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# custom modules

import argparse
import sys
import os
import shutil
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.optimizers import Adam

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py


# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                            opt.minibatch_size, opt.valid_size,
                            opt.states_fil, opt.labels_fil)
# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels./
# train_data = trans.get_train()
# valid_data = trans.get_valid()
#
# alternatively you can get one random mini batch line this
#
# NOT WORKING!
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

img_x = opt.pob_siz*opt.cub_siz
img_y = opt.pob_siz*opt.cub_siz


[x_train, labels_train] = trans.get_train()
print("x train shape before reshape: {}".format(x_train.shape))
print("labels train shape before reshape: {}".format(labels_train.shape))

[x_valid, labels_valid] = trans.get_valid()
print("x valid shape before reshape: {}".format(x_valid.shape))
print("labels valid shape before reshape: {}".format(labels_valid.shape))

x_train = x_train.reshape(x_train.shape[0], img_x,img_y,opt.hist_len)
x_valid = x_valid.reshape(x_valid.shape[0],img_x,img_y,opt.hist_len)
print("x train shape after reshape: {}".format(x_train.shape))
print("x valid shape after reshape: {}".format(x_valid.shape))


input_shape = (img_x, img_y, opt.hist_len)
print("input shape: {}".format(input_shape))

print("x train shape: {}".format(x_train.shape))
print("train samples: {}".format(x_train.shape[0]))
print("test samples: {}".format(x_valid.shape[0]))



x_train = x_train.astype("float32")
x_valid = x_valid.astype("float32")

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu'))


model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(opt.act_num, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()


model.fit(x_train, labels_train,
          batch_size=opt.minibatch_size,
          epochs=opt.eval_nepisodes,
          verbose=1,
          validation_data=(x_valid, labels_valid),
          callbacks=[history])


# 2. save your trained model

model.save("my_agent.h5")
print("model saved!")
