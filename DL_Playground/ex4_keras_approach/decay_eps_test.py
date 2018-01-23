#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

epsilon = 1.0
decay_rate = 0.003
episodes = 3000
const = 0.2
init_eps = epsilon-const
epsilon = []

for i in range(episodes):
    new_eps = init_eps*np.exp(-decay_rate*i)+const
    epsilon.append(new_eps)
    
plt.figure()
plt.plot(epsilon)
plt.show()
