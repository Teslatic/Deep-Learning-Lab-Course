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
from mountain_car import MountainCarEnv

def test_run(agent):
    test_reward_list = []
    for i in range(TESTS):
        test_episode_reward = 0
        env.seed(i)
        state = env.reset()
        state = state.reshape((1,2))
        tmp_epsilon = agent.epsilon
        agent.epsilon = 0
        for _ in range(200):
            if i % SHOW_PROGRESS == 0:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action,'vanilla')
            next_state = next_state.reshape(1,2)
            if done:
                break
            state = next_state
            test_episode_reward += reward
        test_reward_list.append(test_episode_reward)
    avg_test_reward = np.mean(test_reward_list)
    print("\naverage reward: {}".format(avg_test_reward))
    agent.epsilon = tmp_epsilon
    return avg_test_reward


weights_file = "network.h5"
action_space = np.arange(3)
input_shape = 2
BATCH_SIZE = 32


RUNS = 5
TRAINING_EPISODES = 500
TIMESTEPS = 200
AUTO_SAVER = 50
SHOW_PROGRESS = 100
TEST_PROGRESS = 10
TESTS = 50

acc_reward = 0
list_acc_reward = [[] for _ in range(RUNS)]
list_episode_reward = [[] for _ in range(RUNS)]

list_epsilon = [[] for _ in range(RUNS)]
list_avg_reward = [[] for _ in range(RUNS)]
list_time = []
list_acc_reward = []
memory = []

# env = gym.make('MountainCar-v0')
for run in range(RUNS):
    env = MountainCarEnv()
    success = 0
    agent = DankAgent(input_shape,BATCH_SIZE, action_space)
    agent.model.summary()
    memory = []
    # agent.model.load_weights('model_init_weights.h5')
    # agent.target_model.load_weights('target_model_init_weights.h5')
    agent.q_target = np.zeros((BATCH_SIZE,action_space.shape[0]))
    agent.t = np.zeros((BATCH_SIZE,action_space.shape[0]))
    agent.a = np.zeros((BATCH_SIZE,action_space.shape[0]))
    agent.epsilon = 1.0

    for ep in range(TRAINING_EPISODES):
        if (ep+1) % TEST_PROGRESS == 0 and ep != 0:
            list_avg_reward[run].append(test_run(agent))

        acc_reward = 0
        state = env.reset().reshape(1,2)
        for t in range(TIMESTEPS):
            if ep % SHOW_PROGRESS == 0:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1,2)
            acc_reward += reward
            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            if done:
                break
        if t >= 199:
            print("Run {}| Failed to complete in epoch {}| Reward {}| Epsilon {:.2f}| memory length: {}".format(run,ep, acc_reward,agent.epsilon, len(agent.memory)))
        else:
            print("Run {}| Completed in trial {}| Reward {}| Epsilon {:.2f}| memory length: {}".format(run,ep, acc_reward, agent.epsilon, len(agent.memory)))
        list_acc_reward.append(acc_reward)



plt.figure()
plt.plot(list_acc_reward)

plt.figure()
for i in range(RUNS):
    plt.plot(list_avg_reward[i], color='grey')#label='{}'.format(network_setup[network_index]))
plt.plot(np.mean(list_avg_reward,axis=0), label = 'mean', color='red')
plt.plot(np.mean(list_avg_reward,axis=0)+np.std(list_avg_reward,axis=0), label = 'mean+std. dev.',linestyle = '--', color='pink')
plt.plot(np.mean(list_avg_reward,axis=0)-np.std(list_avg_reward,axis=0), label = 'mean-std. dev.',linestyle = '--',color='pink')
plt.title("Avg. reward in a intermediate test every {} episodes".format(TEST_PROGRESS))
plt.xlabel("Test")
plt.ylabel("Reward (Vanilla)")
plt.legend()

plt.show()
