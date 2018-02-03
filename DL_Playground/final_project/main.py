#!/usr/bin/python3

import gym
import sys
import random
from random import randrange
import numpy as np
from DQNAgent import DankAgent
import datetime
import time
import matplotlib.pyplot as plt
import csv
import keras


weights_file = "network.h5"
action_space = np.arange(3)
input_shape = 2
BATCH_SIZE = 32


RUNS = 1
TRAINING_EPISODES = 10000
TIMESTEPS = 500
AUTO_SAVER = 50
SHOW_PROGRESS = 50
TEST_PROGRESS = 10


acc_reward = 0
list_acc_reward = [[] for _ in range(RUNS)]
list_episode_reward = [[] for _ in range(RUNS)]

list_epsilon = [[] for _ in range(RUNS)]
list_avg_reward = [[] for _ in range(RUNS)]
list_time = []

memory = []

env = gym.make('MountainCar-v0')
success = 0
agent = DankAgent(input_shape,BATCH_SIZE, action_space)
agent.model.summary()
for ep in range(TRAINING_EPISODES):

        state = env.reset().reshape(1,2)
        for t in range(TIMESTEPS):
            if ep % SHOW_PROGRESS == 0:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1,2)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            if done:
                break
        if t >= 199:
            print("Failed to complete in epoch {}| Epsilon {:.2f}| memory length: {}".format(ep,agent.epsilon, len(agent.memory)))
        else:
            print("Completed in trial {}| Epsilon {:.2f}| memory length: {}".format(ep, agent.epsilon, len(agent.memory)))
