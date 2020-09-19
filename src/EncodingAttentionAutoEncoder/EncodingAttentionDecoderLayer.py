import tensorflow as tf

from src.Layer.PointWiseFeedForward import PointWiseFeedForward
from src.Layer.GRUGate import GRUGate


class EncodingAttentionDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, rate=0.1):
        super(EncodingAttentionDecoderLayer, self).__init__()

        self.ffn = PointWiseFeedForward(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru1 = GRUGate(d_model)
        self.gru2 = GRUGate(d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, padding_mask):

        in2 = self.layernorm2(x)
        ffn_output = self.ffn(in2)
        ffn_output = tf.keras.activations.relu(ffn_output, alpha=0.0, max_value=None, threshold=0)
        out2 = self.gru2(x, ffn_output)
        out2 = self.dropout2(out2, training=training)

        return out2
