import random
from time import time
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D,Flatten,Dense


class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32,kernel_size=3,strides=2,activation='relu',input_shape= self.state_size))
        model.add(Conv2D(64,kernel_size=3,strides=2,activation='relu'))
        model.add(Conv2D(64,kernel_size=3,strides=2,activation='relu'))
        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(opt.action_size))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(Self.action_size)
        state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
        act_values = model.predict(state)
        return np.argmax(act_values[0])

    def train(self, minibatch, change_epsilon = True):
         state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = minibatch
          for state, action, next_state, reward, done in zip(state_batch, action_batch, next_state_batch, reward_batch, terminal_batch):
              state = state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
                  #reshape the next input frames
                  next_state = next_state.reshape(1,self.state_size[0],self.state_size[1],self.state_size[2])
            else:
                  state = np.array([state])
                  next_state = np.array([next_state])

            action = np.argmax(action)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.model.target_model(next_stae)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            history = self.model.fit(state,target,epochs=1,verbose = 0)

        if self.epsilon > self.epsilon_min and change_epsilon:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)
