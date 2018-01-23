#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from random import randrange
import tensorflow as tf
import keras
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Dropout

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
from astar_demo import astar_solver
from agent import Agent, helper_save, append_to_hist, print_timestamp
import argparse
import sys
#sys.path.append('..')
parser = argparse.ArgumentParser()
import random
import datetime

#################### TESTING AREA ############################




## INITIALIZATION ############################################

# history of total episode rewards
episode_reward_hist = []
#history of total episode step needed to solve or ealy step
epi_step_hist = [] 
epi_step = 0
nepisodes = 0
#sum of a all rewards in one episode
episode_reward = 0 
disp_progress = False


TEST_EPISODES = 200
SHOW_PROGRESS = 1 


opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)
opt.disp = True




## LOAD MODEL ################################################

# creat a test agent
test_agent = Agent()
parser.add_argument("-w", "--weights", help=".h5 weights_file_name for conv network",
                    default=opt.weights_fil)
args = parser.parse_args()
weights_file_name = args.weights
test_agent.load(weights_file_name)

###############################################################
# create astar solver and execute n times initally to generate metric to compare agent
astar = astar_solver(TEST_EPISODES,False)
print("A-Star: {}/{} solved\t| average steps needed: {:.2f}".format(astar.sim.success,TEST_EPISODES,np.mean(astar.list_epi_steps)))

nepisodes_solved_cnt = 0
#termin_state = []

opt.disp = False
if opt.disp_on:
    win_all = None
    win_pob = None
state = sim.newGame(opt.tgt_y, opt.tgt_x)

state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)




while nepisodes < TEST_EPISODES:
    if state.terminal or epi_step >= opt.early_stop: 
        disp_progress = True if nepisodes % SHOW_PROGRESS == 0 else False
        nepisodes += 1
        if state.terminal:
            nepisodes_solved_cnt +=1
           

        print("Tested Episode {}/{}\t| Reward per Episode: {:.2}\t|Steeps needed {}"
                      .format(nepisodes,TEST_EPISODES, episode_reward,epi_step ))
        #termin_state.append(state.terminal)
        epi_step_hist.append(epi_step)
        episode_reward_hist.append(episode_reward)
        episode_reward = 0
        epi_step = 0
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        #astar_steps,state =start_new_game()
        #astar_steps_hist.append(astar_steps)

        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    
    action = test_agent.act(np.array([state_with_history]),False)
    
    epi_step +=1
    next_state = sim.step(action)
    # append to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))

    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    episode_reward += next_state.reward
    state = next_state

    if opt.disp_on and disp_progress:

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


epi_step_hist =np.array(epi_step_hist)

delta_astar = np.mean(astar.list_epi_steps) - np.mean(epi_step_hist)
print("==============================")
print("success rate of the agent: {}".format(nepisodes_solved_cnt/TEST_EPISODES))
print("mean diff to astare if successful {}".format(delta_astar))


ymin, ymax = 0, opt.early_stop


plt.figure()
plt.ylim(0,opt.early_stop+20)
plt.plot(epi_step_hist,label="DQN-Agent")
plt.plot(astar.list_epi_steps,label="A*")
plt.title("Comparison of needed Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()




plt.show()










