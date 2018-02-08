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
from utils import Options
import datetime

class DankAgent():
    def __init__(self, input_shape, batch_size, action_space):
        self.memory = deque(maxlen = 3000)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.action_space = action_space
        self.gamma = 0.99
        self.epsilon = 1.0
        self.init_epsilon = 1.0
        self.eps_decay_rate = 0.000008
        self.decay_const = 0.00
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.cnt = 1
        self.update_target_marker = 1
        self.t = 1
        self.tau = 0.125
        self.train_start = 1000



    def _build_model(self):
        model = Sequential()
        model.add(Dense(24,input_shape = (self.input_shape,),activation = 'relu',))
        model.add(Dense(48, activation = 'relu',))
        model.add(Dense(24, activation = 'relu',))
        model.add(Dense(int(self.action_space.shape[0]), activation = 'linear' ,))
        model.compile(loss = 'mse', optimizer = Nadam(lr = self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, explore = True):
        if explore:
            pick = np.random.choice(['random','greedy'], p = [self.epsilon,1-self.epsilon])
            if pick == 'random':
                action = random.choice(self.action_space)
            else:
                action_values = self.model.predict(state)
                action = np.argmax(action_values)
            if len(self.memory) > self.train_start:
                self.epsilon = 1.0*np.exp(-self.eps_decay_rate * self.t, None)
                self.t += 1
        else:
            action_values = self.model.predict(state)
            action = np.argmax(action_values)

        return action


    def train(self):
        if len(self.memory) < self.train_start:
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

        q_target = self.model.predict(state_batch)

        a = self.model.predict(next_state_batch)

        t = self.target_model.predict(next_state_batch)


        for idx,action in enumerate(action_batch):
            if done_batch[idx]:
                q_target[idx][action] = reward_batch[idx]
            else:
                q_target[idx][action] = reward_batch[idx] + self.gamma * t[idx][np.argmax(a[idx], axis = 0)]



        self.model.fit(state_batch, q_target, batch_size = self.batch_size, epochs = 1, verbose = 0)

        if self.cnt % self.update_target_marker == 0:
            self.update_target_model()
            self.cnt = 0
        self.cnt += 1



    def load(self, file_name):
        self.model.load_weights(file_name)
        print_timestamp()
        print("agent loaded weights from file '{}' ".format(file_name))
        self.update_target_model()

    def save(self, file_name):
        # print_timestamp()
        self.model.save_weights(file_name)
        # print("agent saved weights in file '{}' ".format(file_name))


    def test_run(self, env):
        test_reward_list = []
        test_steps_list = []
        for i in range(Options.TESTS):
            test_episode_reward = 0
            env.seed(i)
            state = env.reset()
            state = state.reshape((1,2))
            tmp_epsilon = self.epsilon
            self.epsilon = 0
            for t in range(Options.TESTSTEPS):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action,'vanilla')
                next_state = next_state.reshape(1,2)
                if done:
                    break
                state = next_state
                test_episode_reward += reward
            test_steps_list.append(t)
            test_reward_list.append(test_episode_reward)
        avg_test_reward = np.mean(test_reward_list)
        avg_test_steps = np.mean(test_steps_list)
        print("\naverage reward: {}".format(avg_test_reward))
        print("average steps: {}".format(avg_test_steps))
        self.epsilon = tmp_epsilon
        return avg_test_reward, avg_test_steps


def print_timestamp(string = ""):
    now = datetime.datetime.now()
    print(string + now.strftime("%Y-%m-%d %H:%M"))
