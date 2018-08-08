import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# one_hot = True로 하면 데이터가 one-hot 형태로 읽힘.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10  # 0~9 digits recognition = 10 classes

# MNIST data image of shape 28 * 28 = 784 (총 픽셀 수)
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# one_hot 형태의 hypothesis와 Y의 값이 같은지 확인.
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))