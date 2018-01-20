import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
import keras
from collections import deque
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras import backend as K
# from keras.layers import Conv2D, Flatten, Dense, Dropout

# custom modules
from utils     import Options, rgb2gray
from astar_simulator import Simulator
from simulator import Simulator
from transitionTable import TransitionTable
import random


# from train_agent import Agent
from astar_demo import astar_solver
import sys
sys.path.append('..')

astar_agent = astar_solver(10)
