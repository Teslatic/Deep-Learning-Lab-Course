#!/usr/bin/env python

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
import tensorflow as tf

# METHODS
##########################################


def conv_layer(input,shape, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="W")
        b = tf.Variable(tf.constant(0.1,shape=[shape[-1]]),name ="B")
        conv = tf.nn.conv2d(input,w, strides=[1,2,2,1], padding="SAME")
        act = tf.nn.relu(conv + b)
        return act

# fully conected layer with relu activation 
def fc_layer(input, shape, name = "fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="W")
        b = tf.Variable(tf.constant(0.1,shape=[shape[-1]]),name ="B")
        act = tf.nn.relu(tf.matmul(input,w)+b)
        return act


# softmax output
def softmax_output(input,shape,name="softmax"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="W")
        b = tf.Variable(tf.constant(0.1,shape=[shape[-1]]),name ="B")
        y_conv = tf.matmul(input,w)+b
        return y_conv

# crossentropy loss function
def xent(input,name="xent"):
    with tf.name_scope(name):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=input))                                                        
    tf.summary.scalar("Cross_entropy",xent)
    return xent

# dropout regularization
def drop_out(input,name="dropout"):
    with tf.name_scope(name):
        keep_prob = tf.placeholder(tf.float32)
        h_drop_out = tf.nn.dropout(input, keep_prob)
        return keep_prob, h_drop_out
    




x  = tf.placeholder(tf.float32,([None,2500]),name="x")
y_ = tf.placeholder(tf.float32,([None,5]),name="labels")

# reshaping
x_image = tf.reshape(x,[-1,50,50,1])

# Layer 1: convolution
h_conv1 = conv_layer(x_image,[3,3,1,32],"conv_1")

# Layer 2: convolution
h_conv2 = conv_layer(h_conv1,[3,3,32,128],"conv_2")

h_conv3 = conv_layer(h_conv2,[3,3,128,256],"conv_2")



s = 7*7*256
# array flattening
h_conv2_flat = tf.reshape(h_conv3, [-1,s])
# Layer 3: fully connected
h_fc1 = fc_layer(h_conv2_flat, [s,1024],"fc_1")

# Dropout Regularization
keep_prob, h_fc1_drop = drop_out(h_fc1)

# Layer 4: Output with Softmax and Cross Entropy Loss
y_conv  = softmax_output(h_fc1_drop,[1024,5],name="softmax")
cross_entropy = xent(y_conv)


correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
# # both train_data and valid_data contain tupes of images and labels
# train_data = trans.get_train()
# valid_data = trans.get_valid()
# 
# alternatively you can get one random mini batch line this
# 
# NOT WORKING!
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################


x_valid,y_valid = trans.get_valid()
learning_rate = 0.1
n_minibatch_updates = opt.n_minibatches
epochs = opt.eval_nepisodes


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
n_samples = 16000
n_batches = n_samples // opt.minibatch_size


list_cost = []
list_train_acc = []
list_test_acc = []
x_batch_old = np.array([])



print("starting training now  ...")
with tf.Session() as sess:        
    print("------------------------------------------------------------------------------")
    # initialize global variables
    sess.run(tf.global_variables_initializer()) 
   
    for episodes in range(opt.eval_nepisodes):

        test_acc = 0
        train_acc = 0
        loss = 0
        for b in range(n_batches):
            x_batch, y_batch = trans.sample_minibatch()
            #print(np.array_equal(x_batch,x_batch_old))
            train_step.run(feed_dict={x: x_batch, y_: y_batch,keep_prob:0.5})
            train_acc += sess.run(accuracy,feed_dict={x:x_batch, y_:y_batch,keep_prob:0.5})
            # accumulated cross entropy loss
            loss +=sess.run(cross_entropy, feed_dict={x:x_batch,y_: y_batch,keep_prob:0.5})
            x_batch_old = x_batch
    
        list_train_acc.append(train_acc/n_batches)
        list_cost.append(loss/n_batches)
    
        test_acc = sess.run(accuracy, feed_dict={x:x_valid, y_:y_valid, keep_prob:1.0})
        list_test_acc.append(test_acc)

        print("Epoch:\t{:d}\nTraining Accuracy:\t{:.4f}\tTesting Accuracy:\t{:.4f}\tLoss:\t{:.4f}".format(episodes+1,train_acc/n_batches,test_acc,loss/n_batches))
        
        
# 2. save your trained model


