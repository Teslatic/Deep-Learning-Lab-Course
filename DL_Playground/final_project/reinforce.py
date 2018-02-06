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
import time
if "../../" not in sys.path:
  sys.path.append("../../")
from mountain_car import MountainCarEnv

"""
* -------------------------------------------------------------------------------
* There are also TODOs in Policy Class!
* -------------------------------------------------------------------------------
"""

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
VALID_ACTIONS = [0, 1, 2]

class Policy():
  def __init__(self):
    self._build_model()

  def _build_model(self):
    """
    Builds the Tensorflow graph.
    """

    self.states_pl = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="states_pl")
    # The TD target value
    self.targets_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="targets_pl")
    # Integer id of which action was selected
    self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions_pl")

    batch_size = tf.shape(self.states_pl)[0]

    self.fc1 = tf.contrib.layers.fully_connected(self.states_pl, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 20, activation_fn=tf.nn.relu,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))
    self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 3, activation_fn=None,
      weights_initializer=tf.random_uniform_initializer(0, 0.5))

    # -----------------------------------------------------------------------
    # TODO: Implement softmax output
    # -----------------------------------------------------------------------
    # softmax prediction
    self.predictions = tf.nn.softmax(self.fc3)

    # Get the predictions for the chosen actions only
    self.gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl

    self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.gather_indices)
    # -----------------------------------------------------------------------
    # TODO: Implement the policy gradient objective. Do not forget to negate
    # -----------------------------------------------------------------------
    # the objective, since the predefined optimizers only minimize in
    # tensorflow.

    self.objective = -tf.log(self.action_predictions)*self.targets_pl

    self.optimizer = tf.train.AdamOptimizer(0.0001)
    self.train_op = self.optimizer.minimize(self.objective)


  def predict(self, sess, s):
    """
    Args:
      sess: TensorFlow session
      states: array of states for which we want to predict the actions.
    Returns:
      The prediction of the output tensor.
    """
    f_dict = {self.states_pl: [s]}


    p = sess.run(self.predictions, f_dict)[0]

    return np.random.choice(VALID_ACTIONS, p=p), p

  def update(self, sess, s, a, y):
    """
    Updates the weights of the neural network, based on its targets, its
    predictions, its loss and its optimizer.

    Args:
      sess: TensorFlow session.
      states: [current_state] or states of batch
      actions: [current_action] or actions of batch
      targets: [current_target] or targets of batch
    """
    feed_dict = { self.states_pl: [s], self.targets_pl: [y], self.actions_pl: [a]}
    sess.run(self.train_op, feed_dict)

    # loss and q_values do not exist here or aren't use
    # return loss, q_values
    return

class BestPolicy(Policy):
  def __init__(self):
    Policy.__init__(self)
    self._associate = self._register_associate()

  def _register_associate(self):
    tf_vars = tf.trainable_variables()
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:total_vars//2]):
      op_holder.append(tf_vars[idx+total_vars//2].assign((var.value())))
    return op_holder

  def update(self, sess):
    for op in self._associate:
      sess.run(op)

def reinforce(sess, env, policy, best_policy, num_episodes, discount_factor=1.0):
  stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

  for i_episode in range(num_episodes):
    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []

    state = env.reset()
    # print([state])

    # +1 for indexing issues
    print("\rgenerate episode {}/{}".format(i_episode+1,num_episodes),end="")
    sys.stdout.flush()
    for t in range(500):
      # -----------------------------------------------------------------------
      # TODO: Implement this
      # -----------------------------------------------------------------------

      action, prob_actions = policy.predict(sess,state)
      next_state, reward, done, _ = env.step(action)
      # some debugging outputs
      # print("policy.predict probabilities: {}".format(prob_actions))
      # print("picked action: {}".format(action_verbose[action]))
      # print("next state: {}|\treward: {:.4f}".format(next_state,reward))

      # build episode tuple
      episode.append((state,action,reward))

      # cumulative reward per episode
      stats.episode_rewards[i_episode] += reward
      stats.episode_lengths[i_episode] = t

      if done:
          break
      state = next_state

    test_return = []
    for t, ep in enumerate(episode):
        # unwrapping the episode tuple
        s = ep[0]
        a = ep[1]
        r = ep[2]

        # calculate total return per time step
        total_return = sum(discount_factor**i * r for i,t in enumerate(episode[t:]))

        # update the policy, target is the total return
        policy.update(sess,s,a,total_return)

  return stats

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig('episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig('reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)


#####################__GLOBALS__#########################
EPISODES = 3000 #default: 3000



#########################################################


if __name__ == "__main__":
  tf.reset_default_graph()
  env = MountainCarEnv()
  p = Policy()
  bp = BestPolicy()
  action_verbose = {0:"left",1:"nothing",2:"right"}

  sess = tf.Session()
  tf.global_variables_initializer().run(session=sess)

  stats = reinforce(sess, env, p, bp, EPISODES)
  success = 0
  plot_episode_stats(stats)
  # save the policy
  saver = tf.train.Saver()
  saver.save(sess, "./policies.ckpt")

  for _ in range(5):
    state = env.reset()
    for i in range(500):
      env.render()
      chosen_action,prob = p.predict(sess, state)
      print("chosen action: {}\t {}".format(action_verbose[chosen_action],prob))
      state, reward, done, _ = env.step(chosen_action)
      if done:
        if reward==10:
            success +=1
        break
print("success rate: {}%".format(success/5 * 100))
