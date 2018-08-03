import tensorflow as tf

# 1. build graph (tensors)
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# 2. feed data and run operations using sess.run(op)
# 3. update variables in the graph

sess = tf.Session()
print("sess.run([node1, node2]): ", sess.run([node1,node2]))
print("sess.run(node3): ", sess.run(node3))

##

# placeholder : 변수

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b    # same as tf.add(a,b)

# feed_dict = { } : 변수의 값을 전달하는 역할

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))


##

# Tensor Ranks, Shapes, Types
# Tensor : data array 라고 이해함.
# 1) Rank : dimension of array - 0,1,2..n 차원. scalar, vector, matrix, ... n-Tensor
# 2) Shape : number of data per element - Shape 가 [D0, D1] 이고 Rank 가 3 이면 [[1,2], [2,3], [3,4]] 이러한 Tensor 을 의미함.
# 3) Type : data type - tf.float32를 주로 씀
