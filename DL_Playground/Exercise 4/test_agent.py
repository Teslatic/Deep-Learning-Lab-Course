from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from random import randrange

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
from collections import defaultdict, namedtuple
import random
import sys
import os
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
    N : minibatch size
    A : number of actions
    Required inputs:
    Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the
            Q value for each action, each output thus is Q(s,a) for one action
            (this is easier to implement than adding the action as an additional input to your network)
    action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                    (e.g. each row contains only one 1)
    Q_s_next: a NxA matrix containing the Q values for the next states.
    best_action_next: a NxA matrix with the best current action for the next state
    reward: a Nx1 matrix containing the reward for the transition
    terminal: a Nx1 matrix indicating whether the next state was a terminal state
    discount: the discount factor
    """
    # calculate: reward + discount * Q(s', a*),
    # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
    target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    target_q = tf.stop_gradient(target_q)
    # calculate: Q(s, a) where a is simply the action taken to get from s to s'
    selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
    loss = tf.reduce_sum(tf.square(selected_q - target_q))
    return loss



class NeuralNetwork():
    def __init__(self):
        self._build_model()


    def _build_model(self):
        '''
        Build TF Graph
        '''

        self.x = tf.placeholder(tf.float32, shape=(None, opt.hist_len*opt.state_siz))
        self.u = tf.placeholder(tf.float32, shape=(None, opt.act_num))
        self.ustar = tf.placeholder(tf.float32, shape=(None, opt.act_num))
        self.xn = tf.placeholder(tf.float32, shape=(None, opt.hist_len*opt.state_siz))
        self.r = tf.placeholder(tf.float32, shape=(None, 1))
        self.term = tf.placeholder(tf.float32, shape=(None, 1))

        # get the output from your network
        self.Q = self.my_network_forward_pass(self.x)
        self.Qn =  self.my_network_forward_pass(self.xn)

        # calculate the loss
        self.loss = Q_loss(self.Q, self.u, self.Qn, self.ustar, self.r, self.term)

        optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.loss)


    def my_network_forward_pass(self,x):
        with tf.variable_scope("DQN",reuse=tf.AUTO_REUSE):
            x_reshaped = tf.reshape(x,shape=[-1,opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len])
            self.layer_1 = tf.contrib.layers.conv2d(x_reshaped,32, kernel_size=3, stride=2, padding='VALID')
            self.layer_2 = tf.contrib.layers.conv2d(self.layer_1,64, kernel_size=3, stride=2, padding='VALID')
            self.layer_3 = tf.contrib.layers.conv2d(self.layer_2, 64, kernel_size=3, stride=2, padding = 'VALID')
            self.layer_3_flat = tf.contrib.layers.flatten(self.layer_3)
            self.layer_4 = tf.contrib.layers.fully_connected(self.layer_3_flat,512, tf.nn.relu)
            self.layer_4_dropout = tf.contrib.layers.dropout(self.layer_4,0.5)
            self.layer_5_output = tf.contrib.layers.fully_connected(self.layer_4_dropout, opt.act_num, activation_fn=None)
            return self.layer_5_output

    # here we predict the action values for the next state
    def predict(self,sess,states):
        feed_dict = {self.xn:states}
        return sess.run(self.Qn,feed_dict)

    # here the network is trained
    def train(self,sess,state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
        loss = sess.run(self.loss, feed_dict = {self.x : state_batch, self.u : action_batch,
                    self.ustar : action_batch_next, self.xn : next_state_batch,
                    self.r : reward_batch, self.term : terminal_batch})

        return loss






def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# You should prepare your network training here. I suggest to put this into a
# class by itself but in general what you want to do is roughly the following
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# setup placeholders for states (x) actions (u) and rewards and terminal values
x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

# get the output from your network
Q = my_network_forward_pass(x)
Qn =  my_network_forward_pass(xn)

# calculate the loss
loss = Q_loss(Q, u, Qn, ustar, r, term)

# setup an optimizer in tensorflow to minimize the loss
"""


print(get_available_devices())

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1000000

EXPLORE = 100000
test_steps = 500

epi_step = 0
nepisodes = 0
INITIAL_EPSILON = 1.0
FINAL_EPSION = 0.1
EPSILON = INITIAL_EPSILON
SHOW_MAP = False





network = NeuralNetwork()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)    

model_number = 2
saver = tf.train.Saver()

# 2. perform a final test of your model and save it
# TODO
print("\nnow performing tests")
with tf.Session() as sess:
    # load model
    saver.restore(sess, "/home/vloeth/Deep-Learning-Lab-Course/DL_Playground/Exercise 4/models/saves_{}/model.ckpt".format(model_number))
    print("model loaded")
    
    if opt.disp_on:
        win_all = None
        win_pob = None

    epi_step = 0
    nepisodes_test = 0
    nepisodes_solved = 0
    action = 0

    # Restart game
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    state_with_history = np.zeros((opt.hist_len, opt.state_siz))
    append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    next_state_with_history = np.copy(state_with_history)

    for step in range(500):

        # Check if episode ended and if yes start new game
        if state.terminal or epi_step >= opt.early_stop:
            epi_step = 0
            nepisodes += 1
            if state.terminal:
                nepisodes_solved += 1
            state = sim.newGame(opt.tgt_y, opt.tgt_x)
            state_with_history[:] = 0
            append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
            next_state_with_history = np.copy(state_with_history)
        else:
            action = np.argmax(network.predict(sess,state_with_history.reshape(1,opt.state_siz*opt.hist_len)))
            state = sim.step(action)
            epi_step += 1


        if opt.disp_on:
            if win_all is None:
                plt.subplot(121)
                win_all = plt.imshow(state.screen)
                plt.subplot(122)
                win_pob = plt.imshow(state.pob)
            else:
                win_all.set_data(state.screen)
                win_pob.set_data(state.pob)
            plt.pause(opt.disp_interval)
    plt.draw()