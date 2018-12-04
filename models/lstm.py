import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicRNNCell

class AveragePooling(object):
  def __init__(self, frame_feature_ph, num_classes, cell_size, use_lstm=False):

    self.frame_feature_ph = frame_feature_ph
    state = tf.reduce_mean(input_tensor=self.frame_feature_ph, axis=1)
    state = tf.nn.relu(state)

    with tf.variable_scope('Classification'):
        self.logit = tf.contrib.layers.fully_connected(inputs=state, num_outputs=num_classes, activation_fn=None) # [batch_size, num_classes]
    self.logit = tf.nn.softmax(self.logit)
    self.prediction = tf.argmax(self.logit, 1)
    self.node = self.prediction
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

class DynamicRNN(object):
  def __init__(self, frame_feature_ph, num_classes, cell_size, use_lstm=False):

    self.frame_feature_ph = frame_feature_ph

    cell = GRUCell(cell_size)
    if use_lstm:
      cell = BasicLSTMCell(cell_size, state_is_tuple=False)
    with tf.variable_scope('DynamicRNN'):
      outputs, state = dynamic_rnn(cell=cell, inputs=self.frame_feature_ph,  dtype=tf.float32)

    outputs = tf.nn.relu(outputs)
    with tf.variable_scope('Classification'):
      node_logit = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=num_classes, activation_fn=None)
    logit = tf.nn.softmax(node_logit)
    self.logit = tf.nn.softmax(tf.reduce_mean(node_logit,1))
    self.node = tf.argmax(logit, 2)
    self.prediction = tf.argmax(self.logit,1)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)
