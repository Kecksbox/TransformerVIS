import tensorflow as tf

from src.PointWiseFeedForward import PointWiseFeedForward
from src.GRUGate import GRUGate


class SelfAttentionParameterLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, rate=0.1):
        super(SelfAttentionParameterLayer, self).__init__()

        self.ffn = PointWiseFeedForward(d_model, dff)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru = GRUGate(d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, static, look_ahead_mask, training):

        in1 = self.layernorm(x)
        in1_extended = tf.concat([in1, static], axis=-1)
        ffn_output = self.ffn(in1_extended, training=training)  # (batch_size, target_seq_len, d_model)
        ffn_output = tf.keras.activations.relu(ffn_output, alpha=0.0, max_value=None, threshold=0)
        out = self.gru(x, ffn_output)  # (batch_size, target_seq_len, d_model)
        out = self.dropout(out, training=training)

        return out
