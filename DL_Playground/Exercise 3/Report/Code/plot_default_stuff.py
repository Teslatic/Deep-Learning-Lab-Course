

import matplotlib.pyplot as plt
import numpy as np
plt.figure()
x, y = np.loadtxt('train_loss_default.csv', delimiter=',', unpack=True)
plt.plot(x,y, label='Train Loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss\nhis_len=4')
plt.legend()

plt.figure()
x, y = np.loadtxt('train_acc_default.csv', delimiter=',', unpack=True)
plt.plot(x,y, label='Train Accuracy')

plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Training Accuracy\nhis_len=4')
plt.legend()

plt.figure()
x, y = np.loadtxt('val_loss_default.csv', delimiter=',', unpack=True)
plt.plot(x,y, label='Val Loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Validation Loss\nhis_len=4')
plt.legend()

plt.figure()
x, y = np.loadtxt('valid_acc_default.csv', delimiter=',', unpack=True)
plt.plot(x,y, label='Val Acc')

plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy\nhis_len=4')
plt.legend()

plt.show()
