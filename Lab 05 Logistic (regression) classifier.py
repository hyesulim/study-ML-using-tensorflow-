# Logistic Regression
# Sigmoid 함수를 이용하여 logistic hypothesis 를 정의함.
# y의 값(1,0)에 따라 다르게 작동하는 cost 함수 정의함. (cost 함수는 실제 데이터와 가설의 차이를 의미함.)
# gradient descent algorithm 사용.
# cost 를 최소화하는 W 를 찾는 것이 학습의 목표임.
# binary classification 을 할 것이므로 y 는 0 또는 1의 값만 가진다.

import tensorflow as tf

x_data = [[1,2], [2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 2]) # multi variable data
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W의 크기는 [들어오는 X의 개수, 나가는 Y의 개수]
# b의 크기는 [나가는 Y의 개수]
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid (logistic hypothesis)
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# gradient descent algorithm 사용.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation(predicted와 실제 Y 값을 비교하여 맞게 예측한 정도를 파악)
# True if hypothesis>0.5 else False
# True, False 를 casting 하면 1, 0으로 형변환됨.
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis:\n", h, "\nCorrect (Y):\n", c, "\nAccuracy: ", a)
