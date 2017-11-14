### import stuff for the tutorials

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import shutil

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.examples.tutorials.mnist import input_data


### CONTROL FLAGS

FLAGS = None
ADAM = False

# weight initialization method, drawn from normal distribution with noise
# to prevent symmetry
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# bias initialization, small bias to prevent "dead neurons"
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

 # convolutional layer with stride = 1 and zero padding -> keeps dimensions!
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

# 2x2 max pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding = "SAME")




learning_rate = [0.1,0.01,0.001,0.0001]
#mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)
	
x  = tf.placeholder(tf.float32,([None,784]))
y_ = tf.placeholder(tf.float32,([None,10]))
	
# reshaping input vector x into 4-dimensional tensor
# 28x28 image with 1 color channel
x_image = tf.reshape(x,[-1,28,28,1])
	
### Layer 1 (convolution)
# flow: input -> convolve with weight -> add bias -> apply ReLu -> max pool
# max pooling will reduce the image sie to 14x14
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])
	
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)







### Layer 2 (convolution)
W_conv2 = weight_variable([7, 7, 16, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

### Layer 3	(fully connected, ReLu activation 128 neurons)
W_fc1 = weight_variable([7*7*16,128])
b_fc1 = bias_variable([128])
	
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# dropout regularization to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Layer 4 (output layer with softmax readout)
W_fc2 = weight_variable([128,10])
b_fc2 = bias_variable([10])
	
y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2


cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
if ADAM == False:
	# define optimizing approach: gradient descent
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(1000):
			batch_xs, batch_ys = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))

			train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if ADAM:
	# use ADAM
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(20000):
			batch = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={
					x: batch[0], y_: batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
				train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

			print('test accuracy %g' % accuracy.eval(feed_dict={
				x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))










