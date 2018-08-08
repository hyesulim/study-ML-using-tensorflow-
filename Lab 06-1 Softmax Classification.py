# Lab 06-1 Softmax Classification
# Softmax : 여러개의 class로 구분해야 하는 경우

# 1. hypothesis
# 2. cost
# 3. gradient descent algorithm

import tensorflow as tf

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
nb_classes = 3  # number of classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# 1. hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)

# 2. Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

# 3. gradient descent algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    all = sess.run(hypothesis, feed_dict={X: [[1,11,7,9], [1,3,4,3], [1,1,0,1]]})
    print(all, sess.run(tf.arg_max(all, 1)))  # 가장 높은 확률을 가지는 argument의 index 출력.