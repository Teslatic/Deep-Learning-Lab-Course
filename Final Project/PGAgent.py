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
from utils import Options

class PolicyApprox():
    def __init__(self):
        self.input_dim = 2
        self.action_dim = 3
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.memory = []
        self.valid_actions = np.arange(self.action_dim)
        self._build_model()

    def _build_model(self):
        '''
        Build Graph
        '''

        # Input
        self.states_pl = tf.placeholder(shape = [None,self.input_dim], dtype=tf.float32, name = 'input')
        # ID of selected action
        self.actions_pl = tf.placeholder(shape = [None], dtype=tf.int32, name = 'actions')
        # target aka return (v_t)
        self.targets_pl = tf.placeholder(shape = [None], dtype=tf.float32, name = 'targets')

        batch_size = tf.shape(self.states_pl)[0]

        self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
          weights_initializer=tf.random_uniform_initializer(0, 0.5))
        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
          weights_initializer=tf.random_uniform_initializer(0, 0.5))
        self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 3, activation_fn=None,
          weights_initializer=tf.random_uniform_initializer(0, 0.5))
        #
        # self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 24, activation_fn = tf.nn.relu)
        #
        # self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 48, activation_fn = tf.nn.relu)
        #
        # self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 24, activation_fn = tf.nn.relu)
        #
        # self.fc4 = tf.contrib.layers.fully_connected(self.fc3, self.action_dim, activation_fn = None)

        # softmax classification
        self.predictions = tf.nn.softmax(self.fc3)

        self.gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl

        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.gather_indices)


        self.objective = -tf.log(self.action_predictions)*self.targets_pl
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.objective)


    def predict(self, sess, s):
        feed_dict = {self.states_pl : [s]}

        p = sess.run(self.predictions, feed_dict)[0]

        action = np.random.choice(self.valid_actions, p=p)

        return action, p


    def remember(self, state, action, reward, done, prob):
        self.memory.append([state, action, reward, done, prob])

        

    def update(self, sess, s, a, v_t):
        feed_dict = { self.states_pl : [s], self.actions_pl : [a], self.targets_pl : [v_t]}
        sess.run(self.train_op, feed_dict)



    def test_run(self, sess,env):
        test_reward_list = []
        test_steps_list = []
        for i in range(Options.TESTS):
            test_episode_reward = 0
            env.seed(i)
            state = env.reset()
            for t in range(Options.TESTSTEPS):
                action,_ = self.predict(sess,state)
                next_state, reward, done, _ = env.step(action,'vanilla')
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
        return avg_test_reward, avg_test_steps
