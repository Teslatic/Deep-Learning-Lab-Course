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
    
    
if opt.map_ind==0:
    reward_threshold = -70
else:
    reward_threshold = -10

################# INIT CUSTOM MODULES AND STUFF ############################
agent = Agent()
agent.model.summary()
#agent.load('plots/rand_start_hist_4.h5')


#number of total trainign game episodes, default: 1000
TRAINING_EPISODES = 1500
AUTO_SAVER = 50

# show a full episode every n episodes for opt.disp_on is true
SHOW_PROGRESS = 50 

# full random episodes before training
RANDOM_EPISODES = 500



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
        
        display_progress = True if nepisodes % SHOW_PROGRESS == 0 else False
        
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
        

        
        
        
        
        
        
        
        
        
        
'''
#### DUMPSTER ####

        
        
        



##################### TESTING AREA ############################

### LOAD MODEL ################################################

## creat a test agent
#test_agent = Agent()
#parser.add_argument("-w", "--weights", help=".h5 weights_file_name for conv network",
                    #default=opt.weights_fil)
#args = parser.parse_args()
#weights_file_name = args.weights
#test_agent.load(weights_file_name)


### INITIALIZATION ############################################

## history of total episode rewards
#episode_reward_hist = []
##history of total episode step needed to solve or ealy step
#epi_step_hist = [] 
#epi_step = 0
#nepisodes = 0
##sum of a all rewards in one episode
#episode_reward = 0 
#disp_progress = False


#N_EPISODES_TOTAL_TEST = 1000
#SHOW_PROGRESS = 1 


################################################################

## create astar solver and execute n times initally to generate metric to compare agent
#astar = astar_solver(N_EPISODES_TOTAL_TEST)
#agent
#print("A-Star: {}/{} solved\t| average steps needed: {:.2f}".format(astar.sim.success,N_EPISODES_TOTAL_TEST,np.mean(astar.list_epi_steps)))

#nepisodes_solved_cnt = 0
#termin_state = []


#state_with_history = np.zeros((opt.hist_len, opt.state_siz))
#append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
#next_state_with_history = np.copy(state_with_history)




#while nepisodes < N_EPISODES_TOTAL_TEST:
    #if state.terminal or epi_step >= opt.early_stop or episode_reward < -10 :
        #disp_progress = True if nepisodes % SHOW_PROGRESS == 0 else False
        #nepisodes += 1
        #if state.terminal:
            #nepisodes_solved_cnt +=1
            #print("nepisodes_solved: {}".format(nepisodes_solved_cnt))

        #print("played {}/{} episodes, episode_reward: {:.2}, epi_step {}"
                      #.format(nepisodes,N_EPISODES_TOTAL_TEST, episode_reward,epi_step ))
        #termin_state.append(state.terminal)
        #epi_step_hist.append(epi_step)
        #episode_reward_hist.append(episode_reward)
        #episode_reward = 0
        #epi_step = 0
        ##astar_steps,state =start_new_game()
        ##astar_steps_hist.append(astar_steps)

        ## and reset the history
        #state_with_history[:] = 0
        #append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        #next_state_with_history = np.copy(state_with_history)

    
    #action = test_agent.act(np.array([state_with_history]),False)
    
    #epi_step +=1
    #next_state = sim.step(action)
    ## append to history
    #append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))

    ## mark next state as current state
    #state_with_history = np.copy(next_state_with_history)
    #episode_reward += next_state.reward
    #state = next_state

    #if opt.disp_on and disp_progress:

        #if win_all is None:
            #plt.subplot(121)
            #win_all = plt.imshow(state.screen)
            #plt.subplot(122)
            #win_pob = plt.imshow(state.pob)
        #else:
            #win_all.set_data(state.screen)
            #win_pob.set_data(state.pob)
        #plt.pause(opt.disp_interval)
        #plt.draw()


#epi_step_hist =np.array(epi_step_hist)

#delta_astar = np.mean(astar.list_epi_steps) - np.mean(epi_step_hist)
#print("==============================")
#print("success rate of the agent: {}%".format((nepisodes_solved_cnt/N_EPISODES_TOTAL_TEST)*100))
#print("mean diff to astare if successful {}".format(delta_astar))













#class Agent:
    #def __init__(self):
        #self.state_size = (opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len)
        #self.action_size = opt.act_num
        #self.gamma = 0.95
        #self.learning_rate = 1e-5
        #self.init_epsilon = 1.0
        #self.final_epsilon = 0.1
        #self.epsilon = self.init_epsilon
        #self.eps_decay_rate = 0.99
        ## default explore value: 100000
        #self.explore = 100000       
        #self.model = self._build_model()
        #self.target_model = self._build_model()
        #self.update_target_model()
        #self.rnd = 1e-18
        #self.greedy = 0        
    
    #def _build_model(self):
        #model = Sequential()
        #model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                     #activation='relu',
                     #input_shape=self.state_size))
        #model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(Flatten())
        #model.add(Dense(512, activation='relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(self.action_size))
        #model.compile(loss='mse',
        #optimizer=Adam(lr=self.learning_rate))
        
        #return model
    
    #def update_target_model(self):
        ## copy weights from model to target_model
        #self.target_model.set_weights(self.model.get_weights())


    #def act(self, state, en_explore = True):        
        ## act randomly during exploration phase
        #if random.random() <= self.epsilon and en_explore:
            ## Random action
            #action = randrange(self.action_size)
            #self.rnd += 1
        ## act greedily after exploration phase
        #else:
            ## reshape state to make convolution work
            #state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            #action_values = self.model.predict(state)
            #action = np.argmax(action_values[0])
            #self.greedy += 1
            
        #return action
    
    #def train(self, minibatch):        
        #state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = minibatch
        #for state, action, next_state, reward, done in zip(state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
            ## reshape input state
            #state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            ## reshape next state
            #next_state = next_state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            
            ## pick best action (update equation -> Q-Learning)
            #action = np.argmax(action)
            ## predict with target model
            #target = self.model.predict(state)
            #if done:
                #target[0][action] = reward
            #else:
                #a = self.model.predict(next_state)[0]
                #t = self.target_model.predict(next_state)[0]
                #target[0][action] = reward + self.gamma * t[np.argmax(a)]
            #history = self.model.fit(state, target, epochs=1, verbose=0)#, callbacks=[tensorboard])
                
            ## decay the epsilon as long it's larger then final epsilon threshold
            #if(self.epsilon>self.final_epsilon):
                #self.epsilon -= ((self.init_epsilon-self.final_epsilon)/self.explore)
                ##self.epsilon *= self.eps_decay_rate

    #def load(self, file_name):
        #self.model.load_weights(file_name)
        #print_timestamp()
        #print("agent loaded weights from file '{}' ".format(file_name))
        #self.update_target_model()
        
    #def save(self, file_name):
        #print_timestamp()
        #self.model.save_weights(file_name)
        #print("agent saved weights in file '{}' ".format(file_name))



'''
