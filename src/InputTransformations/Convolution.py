import tensorflow as tf

from src.Layer.GRUGate import GRUGate


class Convolution(tf.keras.layers.Layer):
    def __init__(self, convolutions, d_model, rate=0.1, time_distributed=True):
        super(Convolution, self).__init__()

        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(800, activation='relu'),
            tf.keras.layers.Dense(d_model, activation='relu')  # shape when flattend in the input convolution
        ])

        self.dropout = tf.keras.layers.Dropout(rate)

    def getTargetShape(self, voxel_shape):
        return ()

    def call(self, x, training):

        out = self.dense(self.flatten(x, training=training), training=training)
        out = self.dropout(out, training=training)

        return out  # (batch_size, input_seq_len, d_model)
