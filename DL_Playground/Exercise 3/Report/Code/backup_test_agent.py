#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from random import randrange
# custom modules
from utils     import Options,rgb2gray
from simulator import Simulator
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D, Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.optimizers import Adam
from transitionTable import TransitionTable

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)

# format state history
state_history = np.zeros((1,opt.pob_siz*opt.cub_siz,opt.pob_siz*opt.cub_siz,opt.hist_len))
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)


# TODO: load your agent
agent = load_model("my_agent.h5")

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
        # this just gets a random action
        # action = randrange(opt.act_num)
        #model.predict()
        gray_state = rgb2gray(state.pob)
        gray_state = gray_state.reshape(1,625)
        trans.add_recent(step, gray_state)
        recent = trans.get_recent()
        print("gray state: {}| shape {}".format(gray_state,gray_state.shape))
        recent_shaped = recent.reshape(1,25,25,opt.hist_len)

        action = agent.predict(recent_shaped)

        state = sim.step(action)
        
      
        #state = sim.step(action)
        print("state history: {}| shape: {}".format(state_history,state_history.shape))
        
        
        
    

        epi_step += 1

    if state.terminal or epi_step >= opt.early_stop:
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
print(float(nepisodes_solved) / float(nepisodes))
# 3. TODO perhaps  do some additional analysis
