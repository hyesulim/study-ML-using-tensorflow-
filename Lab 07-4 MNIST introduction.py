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

# Training epoch/batch
# one epoch : 전체 데이터 셋을 1번 다 학습시킨다
# batch size : the number of training examples in one pass
# 1000 training examples, 500 batch size -> 2 iterations to complete 1 epoch

training_epoch = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoch):
        avg_cost = 0
        # total_batch : number of iterations
        total_batch = int(mnist.train.num_examples / batch_size)

        # 아래 loop가 끝나면 1 epoch이 끝난 것임.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
