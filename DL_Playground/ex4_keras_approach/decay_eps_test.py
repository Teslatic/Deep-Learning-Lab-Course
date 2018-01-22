#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

init_eps = 1.0
decay_rate = 0.003
episodes = 3000

epsilon = []

for i in range(episodes):
    new_eps = init_eps*np.exp(-decay_rate*i)
    epsilon.append(new_eps)
    
plt.figure()
plt.plot(epsilon)
plt.show()
