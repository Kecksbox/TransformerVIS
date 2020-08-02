import tensorflow as tf


class GRUGate(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(GRUGate, self).__init__()

        self.linear_w_r = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        self.linear_u_r = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        self.linear_w_z = tf.keras.layers.Dense(d_model, bias_initializer=tf.constant_initializer(
            value=-2
        ))  ### Giving bias to this layer (will count as b_g so can just initialize negative)
        self.linear_u_z = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        self.linear_w_g = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        self.linear_u_g = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)

    def call(self, x, y, training):
        z = tf.keras.activations.sigmoid(self.linear_w_z(y, training=training) + self.linear_u_z(x, training=training))
        r = tf.keras.activations.sigmoid(self.linear_w_r(y, training=training) + self.linear_u_r(x, training=training))
        h_hat = tf.keras.activations.tanh(
            self.linear_w_g(y, training=training) + self.linear_u_g(tf.math.multiply(r, x), training=training))
        return (1. - z) * x + z * h_hat
