#!/usr/bin/python3
from collections import defaultdict
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import itertools
import pandas as pd
from PIL import Image
from mountain_car import MountainCarEnv
from PGAgent import PolicyApprox
from utils import Options

###############################################################################


class PGTraining():
    def __init__(self):
        self.list_acc_reward = [[] for _ in range(Options.RUNS)]
        self.list_episode_reward = [[] for _ in range(Options.RUNS)]

        self.list_epsilon = [[] for _ in range(Options.RUNS)]
        self.list_avg_reward = [[] for _ in range(Options.RUNS)]
        self.list_avg_steps = [[] for _ in range(Options.RUNS)]
        self.heatmap_memory = []

        self.list_time = []


    def train_PGAgent(self):

        EpisodeStats = namedtuple('Stats',['episode_lengths', 'episode_rewards'])
        stats = EpisodeStats(episode_lengths = np.zeros(Options.TRAINING_EPISODES), episode_rewards = np.zeros(Options.TRAINING_EPISODES))



        for run in range(Options.RUNS):
            tf.reset_default_graph()
            env = MountainCarEnv()
            policy = PolicyApprox()

            sess = tf.Session()
            tf.global_variables_initializer().run(session=sess)

            for i_episode in range(Options.TRAINING_EPISODES):
                if (i_episode+1) % Options.TEST_PROGRESS == 0 and i_episode != 0:
                    tmp_test_reward, tmp_test_steps = policy.test_run(sess, env)
                    self.list_avg_reward[run].append(tmp_test_reward)
                    self.list_avg_steps[run].append(tmp_test_steps)
                acc_reward = 0
                state = env.reset()
                policy.memory = []
                for t in range(Options.TIMESTEPS):
                    if i_episode % Options.SHOW_PROGRESS == 0 and i_episode !=0:
                        env.render()
                    action, prob = policy.predict(sess, state)
                    # print(i_episode, action, prob)
                    next_state, reward, done, _ = env.step(action,'vanilla')
                    acc_reward += reward
                    policy.remember(state, action, reward, done, prob)
                    self.heatmap_memory.append([state[0],state[1],action])


                    if done:
                        break
                    state = next_state

                self.list_acc_reward[run].append(acc_reward)
                if t >= (Options.TIMESTEPS-1):
                    print("PG || Run {}| Failed to complete in epoch {}| Reward {}".format(run,i_episode, acc_reward))
                else:
                    print("PG || Run {}| Completed in trial {}| Reward {} ".format(run,i_episode, acc_reward))
                for t, ep in enumerate(policy.memory):
                    s = ep[0]
                    a = ep[1]
                    r = ep[2]

                    # get return (base function here is the return v_t)
                    v_t = sum(policy.gamma**i * r for i,t in enumerate(policy.memory[t:]))

                    # now make update step based on state, action, and the target v_t
                    policy.update(sess, s, a, v_t)



        means_acc_reward = np.mean(self.list_acc_reward, axis=0)
        std_dev_acc_reward= np.std(self.list_acc_reward, axis=0)
