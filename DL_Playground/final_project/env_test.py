#!/usr/bin/env python
import random
from random import randrange
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import time

EPISODES = 10
TIMESTEPS = 200






games = ['Breakout-v0', 'Frostbite-v0', 'Jamesbond-v0','SpaceInvaders-v0', 'Berzerk-v0', 'MsPacman-v0']

env = gym.make(games[0])


env.reset()
env.render()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(state_size)
print(action_size)
done = False                                    



best_param = None
best_reward = 0



for i_episodes in range(EPISODES):
    state = env.reset()
    next_state, reward, done, _ = env.step(1)
    total_reward = 0
    for t in range(TIMESTEPS):
        env.render()
        action = random.choice(np.arange(action_size))
        next_state, reward, done, _ = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        time.sleep(0.02)
