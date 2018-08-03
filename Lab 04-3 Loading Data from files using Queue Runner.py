import tensorflow as tf
# 파일 하나로 파일 큐를 생성함.
filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

# reader을 정의함.
# binary를 읽는 reader가 될 수도 있는 등 다양함. 현재는 text를 읽도록 정의함.
# 파일 큐에서 key와 value를 읽어오도록하였음.
reader = tf.TextLineReader()
key, value= reader.read(filename_queue)

# Default values, in case of empty columns.
# Also specifies the type of the decoded result.
# 각각 field의 data type을 float라고 0.을 이용해 정의해 주었음.
record_defaults = [[0.], [0.], [0.], [0.]]
# 위에서 읽은 value를 csv로 decode할 것이며, 위에서 정한 data type을 넘겨받음.(record_defaults)
xy = tf.decode_csv(value, record_defaults=record_defaults)

# xy로 읽어온 것을 batch를 이용하여 pump처럼 가져와 읽을 것임.
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[:-1]], batch_size=10)

# placeholders for a tensor that will be always fed
X = tf.placeholder(tf.float32, shape=[None, 3])
