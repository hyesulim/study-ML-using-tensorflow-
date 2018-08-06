import tensorflow as tf
# 파일 하나로 파일 큐를 생성함.
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

# reader을 정의함.
# binary를 읽는 reader가 될 수도 있는 등 다양함. 현재는 text를 읽도록 정의함.
# 파일 큐에서 key와 value를 읽어오도록 하였음.
reader = tf.TextLineReader()
key, value= reader.read(filename_queue)

# Default values, in case of empty columns.
# Also specifies the type of the decoded result.
# 각각 field의 data type을 float라고 0.을 이용해 정의해 주었음.
record_defaults = [[0.], [0.], [0.], [0.]]
# 위에서 읽은 value를 csv로 decode할 것이며, 위에서 정한 data type을 넘겨받음.(record_defaults)
xy = tf.decode_csv(value, record_defaults=record_defaults)

# xy로 읽어온 것을 batch를 이용하여 pump처럼 가져와 읽을 것임.
# 한번에 10개씩 읽을 것임.
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 3]) # [# of instances, 3 features]
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight') # [# of X feature, # of Y]
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    # 파일에서 batch를 읽어옴.
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    # 학습을 시킬 때 batch를 넘겨줌.
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

# shuffle_batch 등 batch의 순서를 섞는 함수 등도 다양하게 존재함.



