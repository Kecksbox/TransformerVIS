import tensorflow as tf

from src.Utilities.PositionalEncoding import positional_encoding
from src.SelfAttentionAutoEncoder.SelfAttentionParameterLayer import SelfAttentionParameterLayer
from src.Layer.SingleHeadAttention import SingleHeadAttention
from src.Layer.GRUGate import GRUGate


class SelfAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model,
                 num_attention_layers, attention_dff,
                 num_decoder_layers, decoder_dff,
                 maximum_position_encoding, rate=0.1):
        super(SelfAttentionDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_decoder_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.sha = SingleHeadAttention(d_model, attention_dff, num_attention_layers, rate)

        self.layernorm_att = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru_att = GRUGate(d_model)

        self.dropout_enc = tf.keras.layers.Dropout(rate)
        self.dropout_att = tf.keras.layers.Dropout(rate)

        self.param_layers = [SelfAttentionParameterLayer(d_model, decoder_dff, rate)
                           for _ in range(self.num_layers)]

    def call(self, x, static, training, look_ahead_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout_enc(x, training=training)

        in1 = self.layernorm_att(x)
        attn, attention_weights['selfattention_decoder_layer_block'] = self.sha(in1, in1, in1, look_ahead_mask, True)
        attention_weights['selfattention_decoder_layer_block'] = attention_weights['selfattention_decoder_layer_block'][:, :, :-1, 1:]
        attn = tf.keras.activations.relu(attn, alpha=0.0, max_value=None, threshold=0)
        out1 = self.gru_att(x, attn)  # (batch_size, input_seq_len, d_model)
        out1 = self.dropout_att(out1, training=training)

        for i in range(self.num_layers):
            out1 = self.param_layers[i](out1, static, look_ahead_mask, training)

        # x.shape == (batch_size, target_seq_len, d_model)
        return out1, attention_weights
