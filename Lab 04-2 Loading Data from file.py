import tensorflow as tf
import numpy as np

''' slicing
# slicing in python list
nums = [0,1,2,3,4]
print(nums)
print(nums[2:4])
print(nums[2:])
print(nums[:2])
print(nums[:])
print(nums[:-1])
nums[2:4] = [8,9]
print(nums)

# Indexing, Slicing, Iterating
# Arrays can be indexed, sliced, iterated much like lists and other sequence types in Python
# As with Python lists, slicing in NumPy can be accomplished with the colon(:) syntax
# Colon instances(:) can be replaced with dots (...)

a = np.array([1,2,3,4,5])

print(a[1:3])
print(a[-1])
a[0:2] = 9
print(a)

b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(b[:-1])
print(b[-1])
print(b[-1, :])
print(b[-1, ...])
print(b[0:2, :])
'''

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1] # 마지막 열 제외하고 모두.
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X : x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))