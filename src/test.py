import tensorflow as tf
import numpy as np

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

result = tf.reduce_sum(tf.abs(xs - ys))

sess = tf.Session()
print(sess.run(result, feed_dict={xs: x_data, ys: y_data}))
print('over')
