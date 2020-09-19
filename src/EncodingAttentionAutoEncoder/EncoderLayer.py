import tensorflow as tf

from src.Layer.PointWiseFeedForward import PointWiseFeedForward
from src.Layer.GRUGate import GRUGate


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dff, d_tar, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.ffn_r = tf.keras.layers.Dense(d_tar, activation=None)
        self.ffn = PointWiseFeedForward(d_tar, dff)

        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru2 = GRUGate(d_tar)

        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        in2 = self.layernorm2(x)
        ffn_r_output = self.ffn_r(x)
        ffn_output = self.ffn(in2)  # (batch_size, input_seq_len, d_model)
        ffn_output = tf.keras.activations.relu(ffn_output, alpha=0.0, max_value=None, threshold=0)
        out2 = self.gru2(ffn_r_output, ffn_output)  # (batch_size, input_seq_len, d_model)
        out2 = self.dropout2(out2, training=training)

        return out2
