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
CHECK_CPU = 0
# adjust amount of epochs, default = 20
EPOCHS = 20
# adjust batch size, default = 50
BATCH_SIZE = 50
N_MINIBATCH_UPDATES = int(55000/BATCH_SIZE)
# default filter depth
FILTER_SIZE = [16]
# list of learning rates, I added 0.7 to emphasize overshooting/divergence during optimising
LEARNING_RATE = [0.7, 0.1,0.01,0.001,0.0001]
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
				train_acc = 0;
				loss = 0;
				print("------------------------------------------------------------------------------")
				# batch loop, 55000 samples, batchsize is 50 -> 1100 training loops per epoch
				for i in range(N_MINIBATCH_UPDATES):
					batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)  
					#writer = tf.summary.FileWriter("output", sess.graph)
					train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
					#writer.close()
					# accumulated training accuracy
					train_acc += sess.run(accuracy,feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
					# accumulated cross entropy loss
					loss +=sess.run(cross_entropy, feed_dict={x:batch_x,y_: batch_y, keep_prob:0.5})
							
				
				list_train_acc[idx].append(train_acc/N_MINIBATCH_UPDATES)
				list_cost[idx].append(loss/N_MINIBATCH_UPDATES)

				test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,
												y_:mnist.test.labels,keep_prob: 1})

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















####################################
### CODE DUMPSTER - IGNORE BELOW ###
####################################





'''
			if i%50==0:
				train_acc, train_summ = sess.run([accuracy, training_summary],
										feed_dict={x:batch_x, y_:batch_y, keep_prob:0.5})
				writer.add_summary(train_summ,i)
				list_train_acc.append(train_acc)
			
				xent, cost_summ = sess.run([cross_entropy, xent_summary], feed_dict={x:batch_x,
																			y_: batch_y, keep_prob:0.5})
				writer.add_summary(cost_summ,i)
				list_cost.append(xent)
			'''
				
		
			#if i%50 == 0:
			#	print('step {}, training accuracy {:.4f}, cost {:.4f}'.format(i, train_acc, xent))
				
			#if epoch % 5 == 0:
		
		
		
		
		
		
'''
if ADAM == False:
	# define optimizing approach: gradient descent
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(2000):
			batch_xs, batch_ys = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))

			train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if ADAM:
	# use ADAM, not as good as SGD !?
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(2000):
			batch_xs, batch_ys = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={
					x: batch_xs, y_: batch_ys, keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
				train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_:
														  mnist.test.labels, keep_prob: 1.0}))

'''









'''


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

### Layer 1 (convolution)
# flow: input -> convolve with weight -> add bias -> apply ReLu -> max pool
# max pooling will reduce the image sie to 14x14
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

### Layer 2 (convolution)
W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

### Layer 3	(fully connected, ReLu activation 128 neurons)
W_fc1 = weight_variable([7*7*16,128])
b_fc1 = bias_variable([128])


#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# dropout regularization to prevent overfitting
#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Layer 4 (output layer with softmax readout)
#W_fc2 = weight_variable([128,10])
#b_fc2 = bias_variable([10])
	
#y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

'''
