#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
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
import argparse
import sys
parser = argparse.ArgumentParser()
import datetime
from agent import Agent

##### GLOBAL HELPER FUNCTIONS ######################################################

def helper_save(plt_file_name):
    if plt_file_name is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(plt_file_name+'.pdf', format='pdf', dpi=1000)
        from matplotlib2tikz import save as tikz_save
        # tikz_save('../report/ex1/plots/test.tex', figureheight='4cm', figurewidth='6cm')
        tikz_save(plt_file_name + ".tikz", figurewidth="\\matplotlibTotikzfigurewidth", figureheight="\\matplotlibTotikzfigureheight",strict=False)

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))






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
    
# reward threshold for early stopping, -10 for map A(1), -70 for map B(0)
if opt.map_ind==0:
    reward_threshold = -70
else:
    reward_threshold = -10

################# INIT CUSTOM MODULES AND STUFF ############################
agent = Agent()
agent.model.summary()
#agent.load('plots/rand_start_hist_4.h5')


#number of total trainign game episodes, default: 1000, 1500 for map B
TRAINING_EPISODES = 4000
AUTO_SAVER = 50

# show a full episode every n episodes for opt.disp_on is true
SHOW_PROGRESS = 50 

# full random episodes before training, 500 for map B
RANDOM_EPISODES = 1000



#############################################################################

## Training Parameter

#history of total episode step needed to solve or early step
epi_step_hist = [] 

# history of total episode rewards
episode_reward_hist = []


epi_step = 0
nepisodes = 0
display_progress = False
#sum of a all rewards in one episode
episode_reward = 0
list_episode_reward = []
list_epsilon = []

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

while nepisodes < TRAINING_EPISODES:
    if state.terminal or epi_step >= opt.early_stop or episode_reward < reward_threshold:
        
        #display_progress = True if nepisodes % SHOW_PROGRESS == 0 else False
        
        if nepisodes % AUTO_SAVER == 0 and nepisodes != 0:
            print_timestamp("saved")
            agent.save(opt.weights_fil)
                       
        # show statistics per episode
        if nepisodes%1==0:
            print("Episode {}/{}\t|Epsilon: {:.2f}\t|e-greedy ratio: {:.2f}\t episode reward: {:.2f}\t| Steps needed: {}".format(nepisodes+1,TRAINING_EPISODES,agent.epsilon,agent.greedy/agent.rnd,episode_reward,epi_step))


        list_epsilon.append(agent.epsilon)
        epi_step_hist.append(epi_step)
        list_episode_reward.append(episode_reward) 
        episode_reward = 0
        epi_step = 0
        
        # update target model
        agent.update_target_model()
        
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    if nepisodes <= RANDOM_EPISODES:
        action = randrange(opt.act_num)
    else:
        action = agent.act(np.array([state_with_history]))
        
    
    epi_step += 1
    next_state = sim.step(action)

    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))


    trans.add(state_with_history.reshape(-1), trans.one_hot_action(action), next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)

    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    episode_reward += next_state.reward
    state = next_state
    
    if nepisodes > RANDOM_EPISODES:
        agent.train(trans.sample_minibatch())
        agent.epsilon = agent.init_epsilon*np.exp(-agent.eps_decay_rate*(nepisodes-RANDOM_EPISODES))
         
         
         #self.init_epsilon = 1.0
        #self.final_epsilon = 0.1
        #self.epsilon = self.init_epsilon
        #self.eps_decay_rate = 0.99
        
    if opt.disp_on and display_progress:
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
        
        
plt.ioff()        
fig = plt.figure()
plt.title("Reward over time")
plt.plot(list_episode_reward, label="Agent")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.savefig('/home/vloeth/Deep-Learning-Lab-Course/DL_Playground/ex4_keras_approach/tmp/test0.png')
plt.close(fig)




fig = plt.figure()
plt.figure()
plt.title("Decaying Epsilon")
plt.plot(list_epsilon, label="Agent")
plt.xlabel("total timesteps")
plt.ylabel("Epsilon")
plt.legend()
plt.savefig('/home/vloeth/Deep-Learning-Lab-Course/DL_Playground/ex4_keras_approach/tmp/test1.png')
plt.close(fig)

#plt.show()
        
