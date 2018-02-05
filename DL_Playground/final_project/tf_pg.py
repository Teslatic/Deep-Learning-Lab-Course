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


        self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn = tf.nn.relu)

        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn = tf.nn.relu)

        self.fc3 = tf.contrib.layers.fully_connected(self.fc2, self.action_dim, activation_fn = None)

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


###############################################################################

TRAINING_EPISODES = 3000
TIMESTEPS = 500
SHOW_PROGRESS = 25
TEST_EPISODES = 20
RUNS = 5




EpisodeStats = namedtuple('Stats',['episode_lengths', 'episode_rewards'])
stats = EpisodeStats(episode_lengths = np.zeros(TRAINING_EPISODES), episode_rewards = np.zeros(TRAINING_EPISODES))



list_acc_reward = [[] for _ in range(RUNS)]

for run in range(RUNS):
    tf.reset_default_graph()
    env = MountainCarEnv()
    policy = PolicyApprox()

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)



    for i_episode in range(TRAINING_EPISODES):
        acc_reward = 0
        state = env.reset()
        policy.memory = []
        for t in range(TIMESTEPS):
            # if i_episode % SHOW_PROGRESS == 0 and i_episode !=0:
            #     env.render()
            action, prob = policy.predict(sess, state)
            # print(i_episode, action, prob)
            next_state, reward, done, _ = env.step(action)
            acc_reward += reward
            policy.remember(state, action, reward, done, prob)

            if done:
                break
            state = next_state
            # if t >= 499:
            #     print("failed epoch {}".format(i_episode))


        list_acc_reward[run].append(acc_reward)
        print("Run: {} | Epoch {} completed after {} steps".format(run,i_episode,t))

        for t, ep in enumerate(policy.memory):
            s = ep[0]
            a = ep[1]
            r = ep[2]

            # get return (base function here is the return v_t)
            v_t = sum(policy.gamma**i * r for i,t in enumerate(policy.memory[t:]))

            # now make update step based on state, action, and the target v_t
            policy.update(sess, s, a, v_t)



means_acc_reward = np.mean(list_acc_reward, axis=0)
std_dev_acc_reward= np.std(list_acc_reward, axis=0)

plt.figure()
for i in range(RUNS):
    plt.plot(list_acc_reward[i], label = 'Run {}'.format(i))
plt.plot(means_acc_reward, label = 'mean', linestyle='-.')
plt.plot(means_acc_reward+std_dev_acc_reward, label = 'mean+std. dev.', linestyle='--',color='pink')
plt.plot(means_acc_reward-std_dev_acc_reward, label = 'mean-std. dev.', linestyle='--',color='pink')
plt.legend()
plt.show()



success = 0
for _ in range(TEST_EPISODES):
    state = env.reset()
    for t in range(500):
        env.render()
        action, prob = policy.predict(sess, state)
        state, reward, done, _ = env.step(action)
        if done:
            if reward==0:
                success += 1
            break

print("success rate: {}%".format(success/TEST_EPISODES * 100))
