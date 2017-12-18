#!/usr/bin/env python3

import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from random import randrange
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator

from keras.models import load_model
import time
from transitionTable import TransitionTable


# 0. initialization
lst_epi_step = []
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

state_history = np.zeros((1,opt.pob_siz*opt.cub_siz,opt.pob_siz*opt.cub_siz,opt.hist_len))
# TODO: load your agent
# Hint: If using standard tensorflow api it helps to write your own model.py  
# file with the network configuration, including a function model.load().
# You can use saver = tf.train.Saver() and saver.restore(sess, filename_cpkt)

directions = {0:"Nop", 1:"Up", 2:"Down", 3:"Left", 4:"Right"}
model = load_model('my_agent.h5')

# 1. control loop
if opt.disp_on:
    win_all = None
    win_pob = None
epi_step = 0    # #steps in current episode
nepisodes = 0   # total #episodes executed
nepisodes_solved = 0
action = 0     # action to take given by the network

# start a new game
state = sim.newGame(opt.tgt_y, opt.tgt_x)
for step in range(opt.eval_steps):

    # check if episode ended
    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
    else:
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would let your agent take its action
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Hint: get the image using rgb2gray(state.pob), append latest image to a history 
        # this just gets a random action
        
        # history construction
        gray_state = rgb2gray(state.pob)
        gray_state = gray_state.reshape(1,opt.state_siz)
        trans.add_recent(step, gray_state)
        recent = trans.get_recent()
        recent_shaped = recent.reshape(1,opt.pob_siz*opt.cub_siz,opt.pob_siz*opt.cub_siz,opt.hist_len)
        
        #random action
        #action = randrange(opt.act_num)
        #state = sim.step(action)
        #print(directions[action])
        
        # trained model prediction
        action = model.predict(recent_shaped)
        print(directions[np.argmax(action)])
        
        state = sim.step(np.argmax(action))
        
        

        
        
        epi_step += 1

    if state.terminal or epi_step >= opt.early_stop:
        print("Steps needed for current episode: {}".format(epi_step))
        lst_epi_step.append(epi_step)
        epi_step = 0
        nepisodes += 1
        if state.terminal:
            nepisodes_solved += 1
        # start a new game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)

    if step % opt.prog_freq == 0:
        print(step)

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

# 2. calculate statistics

print("Steps needed on average to succeed: {:.4f}".format(np.mean(lst_epi_step)))


print(float(nepisodes_solved) / float(nepisodes))
# 3. TODO perhaps  do some additional analysis
