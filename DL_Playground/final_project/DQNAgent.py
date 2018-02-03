import random
from random import randrange
from time import time
import numpy as np
import keras
from collections import deque
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, Nadam
from keras import backend as K
from keras.layers import Flatten,Dense, Dropout
import datetime

class DankAgent():
    def __init__(self, input_shape, batch_size, action_space):
        self.memory = deque(maxlen = 2000)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.action_space = action_space
        self.gamma = 0.9
        self.epsilon = 1.0
        self.init_epsilon = 0.95
        self.eps_decay_rate = 0.000001#45
        self.decay_const = 0.05
        self.learning_rate = 0.005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.cnt = 1
        self.update_target_marker = 1
        self.t = 0
        self.tau = 0.125

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24,input_shape = (self.input_shape,),activation = 'relu',))
        model.add(Dense(48, activation = 'relu',))
        model.add(Dense(24, activation = 'relu',))
        model.add(Dense(int(self.action_space.shape[0]), activation = 'linear' ,))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])


    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau +target_weights[i] *(1 - self.tau)
        self.target_model.set_weights(target_weights)

    def act(self, state):
        pick = np.random.choice(['random','greedy'], p = [self.epsilon,1-self.epsilon])
        if pick == 'random':
            # action = np.where(self.action_space==random.choice(self.action_space))[0]
            action = random.choice(self.action_space)
        else:
            action_values = self.model.predict(state)
            action = np.argmax(action_values)
        self.epsilon = self.init_epsilon*np.exp(-self.eps_decay_rate*self.t)+self.decay_const
        self.t += 1
        return action


    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        state_batch = np.array([x[0] for x in batch])
        state_batch = state_batch.reshape((self.batch_size,self.input_shape))
        action_batch = np.array([x[1] for x in batch])
        action_batch = action_batch.reshape((self.batch_size,1))
        reward_batch = np.array([x[2] for x in batch])
        reward_batch = reward_batch.reshape((self.batch_size,1))
        next_state_batch = np.array([x[3] for x in batch])
        next_state_batch = next_state_batch.reshape((self.batch_size,self.input_shape))
        done_batch = np.array([x[4] for x in batch])
        done_batch = done_batch.reshape((self.batch_size,1))

        # print("state_batch",state_batch)
        # print("action_batch",action_batch)
        # print("reward_batch",reward_batch)
        # print("next_state_batch",next_state_batch)
        # print("done_batch",done_batch)
        q_target = self.model.predict(state_batch)
        # print("q_target")
        # print(q_target)
        # print(q_target.shape)
        a = self.model.predict(next_state_batch)

        t = self.target_model.predict(next_state_batch)


        for idx,action in enumerate(action_batch):
            if done_batch[idx]:
                q_target[idx][action] = reward_batch[idx]
            else:
                q_target[idx][action] = reward_batch[idx] + self.gamma * t[idx][np.argmax(a[idx], axis = 0)]




        # print(np.squeeze(state_batch))
        # for idx,val in enumerate(batch):
        #     state, action, reward, next_state, done = val
        #     print(state)
        #     raise()
        #     target = self.target_model.predict(state)
        #     if done:
        #         target[0][action] = reward
        #     else:
        #         q_next = max(self.target_model.predict(next_state)[0])
        #         print(q_next)
        #         target[0][action] = reward + q_next * self.gamma
        #     self.model.fit(state, target, epochs = 1, verbose=0)
        if self.cnt % self.update_target_marker == 0:
            self.update_target_model()
            self.cnt = 0
        self.cnt += 1

    # def train(self, batch):
    #     state_batch = np.array([x[0] for x in batch])
    #     state_batch = state_batch.reshape((self.batch_size,self.input_shape))
    #     action_batch = np.array([x[1] for x in batch])
    #     action_batch = action_batch.reshape((self.batch_size,1))
    #     reward_batch = np.array([x[2] for x in batch])
    #     reward_batch = reward_batch.reshape((self.batch_size,1))
    #     next_state_batch = np.array([x[3] for x in batch])
    #     next_state_batch = next_state_batch.reshape((self.batch_size,self.input_shape))
    #     done_batch = np.array([x[4] for x in batch])
    #     done_batch = done_batch.reshape((self.batch_size,1))
    #
    #     q_target = self.model.predict(state_batch)
    #
    #
    #     for idx,action in enumerate(action_batch):
    #         if done_batch[idx] == True:
    #             print("done")
    #             q_target[idx][action] = reward_batch[idx]
    #         else:
    #             print("not done")
    #             a = self.model.predict(next_state_batch)
    #             t = self.target_model.predict(next_state_batch)
    #             q_target[idx][action] = reward_batch[idx] + self.gamma * t[idx][np.argmax(a[idx],axis=0)]
    #
    #     self.model.fit(state_batch, q_target, batch_size = self.batch_size, epochs = 1, verbose = 0)
    #
    #     if self.cnt % self.update_target_marker == 0:
    #         self.update_target_model()
    #         self.cnt = 0
    #     self.cnt += 1

    def load(self, file_name):
        self.model.load_weights(file_name)
        print_timestamp()
        print("agent loaded weights from file '{}' ".format(file_name))
        self.update_target_model()

    def save(self, file_name):
        # print_timestamp()
        self.model.save_weights(file_name)
        # print("agent saved weights in file '{}' ".format(file_name))


def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))
