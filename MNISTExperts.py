from tensorflow.examples.tutorials.mnist import input_data

# Dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf

# ---------------------------------------------Beginners----------------------------------------------------------------
# Will let us create computation graphs even after session has been created.
sess = tf.InteractiveSession()

# Placeholders for given info
x = tf.placeholder(tf.float32, [None, 784])  # Features placeholder(Hints)
y_ = tf.placeholder(tf.float32, [None, 10])  # Labels placeholder(Answers)

W = tf.Variable(tf.zeros([784, 10]))  # Weights for single layer(Basic)
b = tf.Variable(tf.zeros([10]))  # Bias for single layer

sess.run(tf.global_variables_initializer())  # Initialize the above two variables

y = tf.matmul(x, W) + b  # To compute the prediction

# Basic loss function using softmax function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # labels = Answers, logits = Predictions
# Actual training using GradientD optimizer(basic) with back propogation to minimize the above cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# training for 1000 epochs
for _ in range(1000):
    batch = mnist.train.next_batch(100)  # batches of 100
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Basically matches the computed predictions with actual answers, argmax = cell with highest value
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # Contains boolean data

# Calculating the mean accuracy. tf.cast = (Changing the boolean data to 1 for True and 0 for False)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # print the following with the given data


# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Experts--------------------------------------------------------

# Using CNN (Convolution layers, activation , pooling , fully connectd layer, final layer)

# Used to create a weight vector of given shape with random values for SYMMETRY BREAKING.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # Random variables
    return tf.Variable(initial)


# Used to create a bias vector of given shape with random values for SYMMETRY BREAKING.
def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Convultion: Extraction of multiple features with a frame of size W with a step_size(strides) of 1 pixel.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Max_pooling: Taking the maximum value in a buffer and then replacing those values with it, frame_size(k_size) = 2x2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])  # Reshaping the flattened images back to their original form 28x28 = 784

# -------------------------------------------------Convolution Layer 1--------------------------------------------------
W_conv1 = weight_variable([5, 5, 1, 32])  # 5x5 matrix, with 1 input but 32 outputs
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(
    conv2d(x_image, W_conv1) + b_conv1)  # Using rectified linear activation function on the convoluted features
h_pool1 = max_pool_2x2(h_conv1)  # Maxpooling on layer 1

# -------------------------------------------------Convolution Layer 2--------------------------------------------------
W_conv2 = weight_variable([5, 5, 32, 64])  # 5x5 matrix, with 32 input but 64 outputs
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(
    conv2d(h_pool1, W_conv2) + b_conv2)  # Using rectified linear activation function on the convoluted features
h_pool2 = max_pool_2x2(h_conv2)  # Maxpooling on layer 2
# -----------------------------------------------Fully Connected Layer 1------------------------------------------------
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 7x7x64 rows, with 1024 neurons
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # Reshaping of Maxpooled layer 2, -1 = Whatever row , 7*7*64 flat
h_fc1 = tf.nn.relu(
    tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # Using rectified linear activation function on the Fully connected layer

keep_prob = tf.placeholder(tf.float32)  # To act as a switch
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # Dropout to prevent overfitting.
# ---------------------------------------------------Final Output Layer-------------------------------------------------
W_fc2 = weight_variable([1024, 10])  # 1024 rows, 10 classes
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # Final predictions
# ----------------------------------------------------------------------------------------------------------------------

# Loss calculation
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)
# Training using the Adam Optimizer(Advanced) with minimizaiton on the loss variable
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Basically matches the computed predictions with actual answers, argmax = cell with highest value
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # Contains boolean data

# Calculating the mean accuracy. tf.cast = (Changing the boolean data to 1 for True and 0 for False)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize the above two variables
    for i in range(20000):  # 20000 Epochs
        batch = mnist.train.next_batch(50)  # Batches of 50
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))  # Print accuracy for every 100th example
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # Actual training
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))  # Print accuracy for the evaluation set
