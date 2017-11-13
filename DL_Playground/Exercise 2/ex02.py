### Quick Tensorflow Introduction ###
'''
import tensorflow as tf

tf.InteractiveSession()
learning_rate = 0.01
x = tf.placeholder(tf.float32)
W = tf.Variable([.3],dtype = tf.float32)
b = tf.Variable([-.3],dtype= tf.float32)

# model
linear_model = W*x + b

#labels
y = tf.placeholder(tf.float32)

# loss
# least squared loss
loss = tf.reduce_sum(tf.square(linear_model - y))
# generate optimizer object with learning rate as argument
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# assign minimizer method from object optimizer to variable train
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) # reset values to wrong
# setup tensorboard
sess.initialize(logdir="log")

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    
accur_W, accur_b, accur_loss = sess.run([W,b,loss],{x: x_train, y:y_train})
print("W: {}".format(accur_W))
print("b: {}".format(accur_b))
print("Loss: {}".format(accur_loss))

'''

### import stuff for the tutorials
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.examples.tutorials.mnist import input_data





### BASIC TUTORIAL FOR TENSORFLOW WITH THE MNIST DATASET


# set learning rate, default 0.5
learning_rate = 0.5

# load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# setup placeholder for input data that is a flattened vector with dimension 784
x = tf.placeholder(tf.float32, [None,784])

# define placeholder for output prediction
y_ = tf.placeholder(tf.float32, [None,10])

# initialze weights with zeros, primitive initialisation
W = tf.Variable(tf.zeros([784,10]))

# initialize bias to 0
b = tf.Variable(tf.zeros([10]))

# define a small epsilon to make logarithm in cross_entropy numerically stable
eps = 1e-10


# setup output nonlinearity: softmax
y = tf.nn.softmax(tf.matmul(x,W)+b)
# alternativeley to adding epsilon, use stable built in function
# y = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x,W)+b)

# define loss function
# 1. tf.log cmoputes logarithm of each element of y
# 2. tf.reduce_sum adds the elements in the second dimension of y
# 3. tf.reduce_mean computes mean over all examples in the batch
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+eps), reduction_indices=[1]))

# define optimizing approach: gradient descent
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# launch the model in an InteractiveSession:
sess = tf.InteractiveSession()

# initialize global variables that are used during training session
tf.global_variables_initializer().run()

# loop training
# gradient descent is now extended to SGD with the batches of size 100
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

# test the trained network on testing data
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy on test set: {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))




###### ADVANCED TENSORFLOW MNIST TUTORIAL



















