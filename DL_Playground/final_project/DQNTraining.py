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
from utils import Options


class DQNTraining():
    def __init__(self):
        self.list_acc_reward = [[] for _ in range(Options.RUNS)]
        self.list_episode_reward = [[] for _ in range(Options.RUNS)]

        self.list_epsilon = [[] for _ in range(Options.RUNS)]
        self.list_avg_reward = [[] for _ in range(Options.RUNS)]
        self.list_avg_steps = [[] for _ in range(Options.RUNS)]
        self.list_time = []
        self.action_space = np.arange(3)
        self.input_shape = 2
        self._generate_init_weights()


    def _generate_init_weights(self):
        self.init_weights = DankAgent(self.input_shape,Options.BATCH_SIZE, self.action_space)
        self.init_weights.model.save_weights('model_init_weights.h5')
        del self.init_weights


    def train_DQNAgent(self):

        weights_file = "network.h5"

        Options.BATCH_SIZE = 32

        acc_reward = 0

        memory = []

        for run in range(Options.RUNS):
            env = MountainCarEnv()
            success = 0
            agent = DankAgent(self.input_shape,Options.BATCH_SIZE, self.action_space)
            agent.model.load_weights('model_init_weights.h5')
            agent.model.summary()
            memory = []

            for ep in range(Options.TRAINING_EPISODES):
                if (ep+1) % Options.TEST_PROGRESS == 0 and ep != 0:
                    # self.list_avg_reward[run].append(agent.test_run(env))
                    tmp_test_reward, tmp_test_steps = agent.test_run(env)
                    # self.list_avg_reward[run].append(policy.test_run(sess, env))
                    self.list_avg_reward[run].append(tmp_test_reward)
                    self.list_avg_steps[run].append(tmp_test_steps)
                acc_reward = 0
                state = env.reset().reshape(1,2)
                for t in range(Options.TIMESTEPS):
                    if ep % Options.SHOW_PROGRESS == 0 and ep != 0:
                        env.render()
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action, 'vanilla')
                    next_state = next_state.reshape(1,2)
                    acc_reward += reward
                    agent.remember(state, action, reward, next_state, done)
                    agent.train()

                    state = next_state
                    if done:
                        break
                if t >= (Options.TIMESTEPS-1):
                    print("DQN || Run {}| Failed to complete in epoch {}| Reward {}| Epsilon {:.2f}| memory length: {}".format(run,ep, acc_reward,agent.epsilon, len(agent.memory)))
                else:
                    print("DQN || Run {}| Completed in trial {}| Reward {}| Epsilon {:.2f}| memory length: {}".format(run,ep, acc_reward, agent.epsilon, len(agent.memory)))
                self.list_acc_reward.append(acc_reward)


        #
        # plt.figure()
        # plt.plot(self.list_acc_reward)
        #
        # plt.figure()
        # for i in range(Options.RUNS):
        #     plt.plot(self.list_avg_reward[i], color='grey')#label='{}'.format(network_setup[network_index]))
        # plt.plot(np.mean(self.list_avg_reward,axis=0), label = 'mean', color='red')
        # plt.plot(np.mean(self.list_avg_reward,axis=0)+np.std(self.list_avg_reward,axis=0), label = 'mean+std. dev.',linestyle = '--', color='pink')
        # plt.plot(np.mean(self.list_avg_reward,axis=0)-np.std(self.list_avg_reward,axis=0), label = 'mean-std. dev.',linestyle = '--',color='pink')
        # plt.title("Avg. reward in a intermediate test every {} episodes".format(Options.TEST_PROGRESS))
        # plt.xlabel("Test")
        # plt.ylabel("Reward (Vanilla)")
        # plt.legend()
        #
        # plt.show()
