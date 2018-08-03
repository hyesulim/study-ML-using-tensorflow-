import tensorflow as tf

# Linear Regression model 구현하기
# H(x) = Wx + b
# cost(W,b) = mean(square(H(xi) - yi))
# Linear Regression 학습의 목표는 cost(W,b)를 최소화 하는 W, b의 값을 구하는 것.

'''1) X, Y의 값을 직접 정해줄 경우'''

'''

# X and Y data (training data set)
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# tf.Variable : trainable varialble. tf가 자체적으로 변경시키는 변수. 학습하는 과정에서 값이 변경되는 변수.
# 처음에는 random 값을 줌.
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis = XW + b
hypothesis = x_train * W + b

# cost/loss function
# tf.reduce_mean(t) : tensor인 t의 평균값을 구해줌
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Gradient Descent
# Minimize
# tf.Variable인 W, b의 값을 조절하여 cost를 minimize 함.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph - tf.Variable 을 사용하기 위해서는 해줘야 함.
sess.run(tf.global_variables_initializer())

# fit the line
# train과 cost, hypothesis, W, b 가 하나의 그래프로 연결이 되어 있으므로, train 을 실행시키면 결국 W, b의 값이 변화함.
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

'''

'''2) placeholder을 이용하여  linear regression model을 만드는 경우'''

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = W*X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0 :
        print(step, cost_val, W_val, b_val)


print(sess.run(hypothesis, feed_dict={X: [5]}))



