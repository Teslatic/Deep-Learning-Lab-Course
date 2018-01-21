from time import time
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D,Flatten,Dense, Dropout

from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
import argparse
import sys
#sys.path.append('..')
parser = argparse.ArgumentParser()
import random
from random import randrange
import datetime


class Agent:
    def __init__(self):
        opt = Options()
        self.state_size = (opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len)
        self.action_size = opt.act_num
        self.gamma = 0.95
        self.learning_rate = 1e-5
        self.init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon = self.init_epsilon
        self.eps_decay_rate = 0.99
        # default explore value: 100000
        self.explore = 1000000       
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.rnd = 1e-18
        self.greedy = 0        
    
    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                     activation='relu',
                     input_shape=self.state_size))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size))
        model.compile(loss='mse',
        optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())


    def act(self, state, en_explore = True):        
        # act randomly during exploration phase
        if random.random() <= self.epsilon and en_explore:
            # Random action
            action = randrange(self.action_size)
            self.rnd += 1
        # act greedily after exploration phase
        else:
            # reshape state to make convolution work
            state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            action_values = self.model.predict(state)
            action = np.argmax(action_values[0])
            self.greedy += 1
            
        return action
    
    def train(self, minibatch):        
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = minibatch
        for state, action, next_state, reward, done in zip(state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
            # reshape input state
            state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            # reshape next state
            next_state = next_state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            
            # pick best action (update equation -> Q-Learning)
            action = np.argmax(action)
            # predict with target model
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            history = self.model.fit(state, target, epochs=1, verbose=0)#, callbacks=[tensorboard])
                
            # decay the epsilon as long it's larger then final epsilon threshold
            if(self.epsilon>self.final_epsilon):
                self.epsilon -= ((self.init_epsilon-self.final_epsilon)/self.explore)
                #self.epsilon *= self.eps_decay_rate
                
    def load(self, file_name):
        self.model.load_weights(file_name)
        print_timestamp()
        print("agent loaded weights from file '{}' ".format(file_name))
        self.update_target_model()
        
    def save(self, file_name):
        print_timestamp()
        self.model.save_weights(file_name)
        print("agent saved weights in file '{}' ".format(file_name))



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
