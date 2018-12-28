import tensorflow as tf
import numpy as np

class rank_model(object):
    def __init__(self, data_size, unit_nums, activate_function=None):
        self.data_size = data_size
        self.unit_nums = unit_nums
        self.activate_function = activate_function
        self.xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self._get_weight()
        self._get_biases()


    def _first_layer(self):
        wx_plus_b = tf.matmul(self.xs, self.weight['in']) + self.biases['in']
        if self.activate_function is None:
            self.first_layer_outputs = wx_plus_b
        else:
            self.first_layer_outputs = self.activate_function(wx_plus_b)

    def _second_layer(self):
        wx_plus_b = tf.matmul(self.first_layer_outputs, self.weight['out']) + self.biases['out']
        self.outputs = wx_plus_b

    def _compute_loss(self):
        loss = 2*

    def _get_weight(self):
        self.weight = {
            'in': tf.Variable(tf.random_normal(shape=[self.data_size, self.unit_nums])),
            'out': tf.Variable(tf.random_normal(shape=[self.unit_nums, self.data_size]))
        }

    def _get_biases(self):
        self.biases = {
            'in': tf.Variable(tf.random_normal(shape=[self.unit_nums])),
            'out': tf.Variable(tf.random_normal(shape=[self.data_size]))
        }
