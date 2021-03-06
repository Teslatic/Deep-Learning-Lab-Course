#!/usr/bin/env python

### import stuff for the tutorials

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import shutil
import numpy as np
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.examples.tutorials.mnist import input_data

### METHODS AND FUNCTIONS ###

def conv_layer(input,shape, name="conv"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="W")
		b = tf.Variable(tf.constant(0.1,shape=[shape[-1]]),name ="B")
		conv = tf.nn.conv2d(input,w, strides=[1,1,1,1], padding="SAME")
		act = tf.nn.relu(conv + b)
		return act

# 2x2 max pooling
def max_pool_2x2(input,name="max_pooling_2x2"):
	with tf.name_scope(name):
		return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],padding = "SAME")


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
		xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
																		logits=input))
	tf.summary.scalar("Cross_entropy",xent)
	return xent
# dropout regularization
def drop_out(input,name="dropout"):
	with tf.name_scope(name):
		keep_prob = tf.placeholder(tf.float32)
		h_drop_out = tf.nn.dropout(input, keep_prob)
		
		return keep_prob, h_drop_out
# plotting stuff
def plot_training_curve(data):
	plt.figure()
	for i,val in enumerate(data):
		plt.plot(data[i],label="learning rate: {}".format(LEARNING_RATE[i]))
	plt.title("Average Training Accuracy per Epoch, Filter depth: {:d}".format(filter_depth))
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()
	return

def plot_testing_curve(data):
	plt.figure()
	for i,val in enumerate(data):
		plt.plot(data[i],label="learning rate: {}".format(LEARNING_RATE[i]))
	plt.title("Testing Accuracy per Epoch, Filter depth: {:d}".format(filter_depth))
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()

def plot_cost_curve(data):
	plt.figure()
	for i,val in enumerate(list_cost):
		plt.plot(data[i],label="learning rate: {}".format(LEARNING_RATE[i]))
	plt.title("Average Cost per Epoch, Filter depth: {:d}".format(filter_depth))
	plt.xlabel("Epoch")
	plt.ylabel("Cost")
	plt.legend()
	
def plott_scatter_runtime(runtime,y):
	plt.figure()
	plt.scatter(y,runtime)
	plt.title("Runtime vs. Number of Parameters with learning rate {:.4f}".format(LEARNING_RATE[0]))	
	plt.ylabel("Runtime in minutes")
	plt.xlabel("Number of parameters")


### MAIN ####

### CONTROL FLAGS
FLAGS = None
# to check GPU/CPU runtime performance, use the according virtual environment and change 
# flags CHECK_CPU before running!
# set to 1 to train with various filter depths and check according runtime performance
CHECK_FILTER_DEPTHS = 1
# set to 1 to run with CPU settings: filter depths are modified (see below)
CHECK_CPU = 1
# adjust amount of epochs, default = 20
EPOCHS = 20
# adjust batch size, default = 50
BATCH_SIZE = 50
N_MINIBATCH_UPDATES = int(55000/BATCH_SIZE)
# default filter depth
FILTER_SIZE = [16]
# list of learning rates, I added 0.7 to emphasize overshooting/divergence during optimising
LEARNING_RATE = [0.7, 0.1,0.01,0.001,0.0001]

# testing batch size for preventing memory problems with GPU -> WORKS!
TEST_BATCH_SIZE = 500
TEST_BATCH_UPDATES = int(10000/BATCH_SIZE)


# filter sizes for experimentation 
# GPU {8, 16, 32, 64, 128, 256}
# CPU {8, 16, 32, 64}
if CHECK_FILTER_DEPTHS:
	if CHECK_CPU:
		FILTER_SIZE = [8, 16, 32, 64]
		# choose fixed learning rate for checking filter depth performance
		LEARNING_RATE = [0.1]
	if CHECK_CPU==0:
		FILTER_SIZE = [8, 16, 32, 64, 128, 256]

		# choose fixed learning rate for checking filter depth performance
		LEARNING_RATE = [0.1]

INPUT_DEPTH = 1
FILTER_WIDTH = 3
PARAMETER_NUM = (FILTER_WIDTH*FILTER_WIDTH*INPUT_DEPTH+1)*FILTER_SIZE

# setup data 55000 training samples, 10000 test samples
print("loading data...")
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
print("...loading finished!")
# global lists
list_runtime = []
list_free_param = []



for index,filter_depth in enumerate(FILTER_SIZE):
	list_cost = [[] for i in range(len(LEARNING_RATE))]
	list_train_acc = [[] for i in range(len(LEARNING_RATE))]
	list_test_acc = [[] for i in range(len(LEARNING_RATE))]	

	x  = tf.placeholder(tf.float32,([None,784]),name="x")
	y_ = tf.placeholder(tf.float32,([None,10]),name="labels")

	# reshaping
	x_image = tf.reshape(x,[-1,28,28,1])

	# Layer 1: convolution
	h_conv1 = conv_layer(x_image,[3,3,1,filter_depth],"conv_1")
	h_pool1 = max_pool_2x2(h_conv1)

	# Layer 2: convolution
	h_conv2 = conv_layer(h_pool1,[3,3,filter_depth,filter_depth],"conv_2")
	h_pool2 = max_pool_2x2(h_conv2)


	# array flattening
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*filter_depth])
	# Layer 3: fully connected
	h_fc1 = fc_layer(h_pool2_flat, [7*7*filter_depth,128],"fc_1")

	# Dropout Regularization
	keep_prob, h_fc1_drop = drop_out(h_fc1)

	# Layer 4: Output with Softmax and Cross Entropy Loss
	y_conv  = softmax_output(h_fc1_drop,[128,10],name="softmax")
	cross_entropy = xent(y_conv)

	correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("starting training now with filter depth of {:d} ...".format(filter_depth))
	start_time = time.time()
	for idx,val in enumerate(LEARNING_RATE):
		# adjust learning rate
		train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE[idx]).minimize(cross_entropy)
		# open new session for every learning rate
		
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with tf.Session(config = config) as sess:
			
			print("------------------------------------------------------------------------------")
			
			# initialize global variables
			sess.run(tf.global_variables_initializer())
			
			# epoch loop
			for ep in range(EPOCHS):
				test_acc = 0;
				train_acc = 0;
				loss = 0;
				print("------------------------------------------------------------------------------")
				# batch loop, 55000 samples, batchsize is 50 -> 1100 training loops per epoch
				for i in range(N_MINIBATCH_UPDATES):
					batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE, shuffle=False)  
					#writer = tf.summary.FileWriter("output", sess.graph)
					train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
					#writer.close()
					# accumulated training accuracy
					train_acc += sess.run(accuracy,feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
					# accumulated cross entropy loss
					loss +=sess.run(cross_entropy, feed_dict={x:batch_x,y_: batch_y, keep_prob:0.5})
							
				
				list_train_acc[idx].append(train_acc/N_MINIBATCH_UPDATES)
				list_cost[idx].append(loss/N_MINIBATCH_UPDATES)

			
				# batching the testing operation in order to prevent OOM issues with GPU support
				for j in range(TEST_BATCH_UPDATES):
					test_batch_x, test_batch_y = mnist.test.next_batch(TEST_BATCH_SIZE)
					test_acc += sess.run(accuracy, feed_dict={x:test_batch_x, y_:test_batch_y,keep_prob: 1})
				test_acc = test_acc/TEST_BATCH_UPDATES
				

				list_test_acc[idx].append(test_acc)
				print("Filter depth:\t\t{:d}\tLearning Rate:\t\t{:.4f}\tEpoch:\t{:d}\nTraining Accuracy:\t{:.4f}\tTesting Accuracy:\t{:.4f}\tCost:\t{:.4f}".format(filter_depth,val,ep,train_acc/N_MINIBATCH_UPDATES,test_acc,loss/N_MINIBATCH_UPDATES))

			
			print("------------------------------------------------------------------------------")
		
		if(CHECK_FILTER_DEPTHS == 1):
			trainable_variables = []
			trainable_variables = tf.trainable_variables()
			free_param = []
			for idx,_ in enumerate(trainable_variables):
				free_param.append(np.prod(trainable_variables[idx].get_shape().as_list()))
			free_param = np.sum(free_param)
			list_free_param.append(free_param)
			print("Trainable parameters: {}".format(free_param))
			tf.reset_default_graph()
			
		
		end_time = time.time()
		time_needed = (end_time - start_time)/60
		list_runtime.append(time_needed)
		print("Training with Filter depth {:d} and learning rate {:.4f} done after {:.4f} min".format(filter_depth,val,time_needed))
		print("------------------------------------------------------------------------------")
		print("------------------------------------------------------------------------------")
	


	# only plot learning curves for comparison of learning rates
	if CHECK_FILTER_DEPTHS==0:
		plot_training_curve(list_train_acc)
		plot_testing_curve(list_test_acc)
		plot_cost_curve(list_cost)
if CHECK_FILTER_DEPTHS == 1:
	plott_scatter_runtime(list_runtime,list_free_param)
plt.show()







