# Lab 06-2 Fancy Softmax Classification

import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])   # 0 ~ 6 총 7개의 정수.
nb_classes = 7

# 주어진 데이터가 one-hot 형태가 아닌 경우
# Y가 0~6의 정수로 표현되므로 이를 3 이 아닌 0001000 형태로 바꾸는 것이 one_hot
Y_one_hot = tf.one_hot(Y, nb_classes)  # one_hot을 하면 차원이 1 높아짐.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # reshape을 통해서 원하는 형태로 데이터를 변형시킴.

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Lab 06-1 에서 사용한 cost. 함수를 사용하지 않고 그대로 식을 풀어 냈음.
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

# softmax_cross_entropy_with_logits
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)   # probability -> 0~6 사이의 값.
# tf.argmax(Y_one_hot, 1) 은 one-hot으로 만들 기 전의 Y와 동일.
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy],
                                 feed_dict={X: x_data, Y:y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))