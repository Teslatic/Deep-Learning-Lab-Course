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
import random



class Agent:
    def __init__(self):
        self.state_size = (opt.cub_siz*opt.pob_siz,opt.cub_siz*opt.pob_siz,opt.hist_len)
        self.action_size = opt.act_num
        self.gamma = 0.95
        self.learning_rate = 1e-5
        self.init_epsilon = 0.9
        self.final_epsilon = 0.1
        self.epsilon = self.init_epsilon
        self.explore = 500000       
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

    def load(self, file_name):
        self.model.load_weights(file_name)
        print("agent loaded weights"+ file_name)
        self.update_target_model()
        
    def save(self, file_name):
        self.model.save_weights(file_name)
        print("agent saved weights"+ file_name)



def helper_save(plt_file_name):
    if plt_file_name is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(plt_file_name+'.pdf', format='pdf', dpi=1000)
        from matplotlib2tikz import save as tikz_save
        # tikz_save('../report/ex1/plots/test.tex', figureheight='4cm', figurewidth='6cm')
        tikz_save(plt_file_name + ".tikz", figurewidth="\\matplotlibTotikzfigurewidth", figureheight="\\matplotlibTotikzfigureheight",strict=False)


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

################# INIT CUSTOM MODULES AND STUFF ############################
agent = Agent()
agent.model.summary()
#agent.load('plots/rand_start_hist_4.h5')


#number of total trainign game episodes
N_EPISODES_TOTAL_TRAIN = 1000 
SAVE_AFTER_N_EPISODES = 50

# show a full episode every n episodes for opt.disp_on is true
DISP_PROGRESS_AFTER_N_EPISODES = 5 

# full random episodes before training
FULL_RANDOM_EPISODES = 50



#############################################################################

## Training Parameter

#history of total episode step needed to solve or ealy step
epi_step_hist = [] 

# history of total episode rewards
episode_reward_hist = []

steps = 1 * 10**6
epi_step = 0
nepisodes = 0
display_progress = False
#sum of a all rewards in one episode
episode_reward = 0 


state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

while nepisodes < N_EPISODES_TOTAL_TRAIN:
    if state.terminal or epi_step >= opt.early_stop or episode_reward < -10:
        
        display_progress = True if nepisodes % DISP_PROGRESS_AFTER_N_EPISODES == 0 else False
        
        if nepisodes % SAVE_AFTER_N_EPISODES == 0 and nepisodes != 0:
            print("saved")
            agent.save("save/" + opt.weights_fil)
                       
        #print("played {}/{} episodes, episode_reward: {:.2}, epi_step {}, epsilon: {:.2}".format(nepisodes,N_EPISODES_TOTAL_TRAIN, episode_reward,epi_step, agent.epsilon))#
        
        if nepisodes%1==0:
            print("Episode {}/{}\t|Epsilon: {:.2f}\t|Epsilon greedy ratio: {:.2f}".format(nepisodes,N_EPISODES_TOTAL_TRAIN,agent.epsilon,agent.greedy/agent.rnd))

        epi_step_hist.append(epi_step)
        episode_reward = 0
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)

    if nepisodes <= FULL_RANDOM_EPISODES:
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
    
    if nepisodes > FULL_RANDOM_EPISODES:
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

    
        
    
f, axarr = plt.subplots(2,1)

axarr[0].plot(episode_reward_hist)
axarr[0].set_ylabel(r'Total Reward',usetex=True)

axarr[1].plot(epi_step_hist)
axarr[1].set_ylabel(r'Number of steps',usetex=True)

axarr[0].set_xlabel(r'Episode',usetex=True)
axarr[1].set_xlabel(r'Episode',usetex=True)    
    #state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()

helper_save(None)

# 2. perform a final test of your model and save it
# TODO
