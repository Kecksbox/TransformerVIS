import tensorflow as tf

from src.Layer.GRUGate import GRUGate


class Convolution(tf.keras.layers.Layer):
    def __init__(self, convolutions, d_model, rate=0.1, time_distributed=True):
        super(Convolution, self).__init__()

        self.dense_r = tf.keras.layers.Dense(d_model, activation=None)
        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

        self.conv_layers = []
        if time_distributed:
            self.flatten_r = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
            self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

            for conv in convolutions:
                self.conv_layers.append(
                    tf.keras.layers.TimeDistributed(
                        tf.keras.layers.Conv3D(
                            filters=conv['filters'],
                            kernel_size=conv['kernel_size'],
                            strides=conv['strides'],
                            activation='relu',
                        )
                    )
                )
        else:
            self.flatten_r = tf.keras.layers.Flatten()
            self.flatten = tf.keras.layers.Flatten()

            for conv in convolutions:
                self.conv_layers.append(
                    tf.keras.layers.Conv3D(
                        filters=conv['filters'],
                        kernel_size=conv['kernel_size'],
                        strides=conv['strides'],
                        activation='relu',
                    )
                )

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.gru = GRUGate(d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        residual = self.dense_r(self.flatten_r(x, training=training))

        inp = self.layernorm(x)

        convolution = inp
        for conv_layer in self.conv_layers:
            convolution = conv_layer(convolution, training=training)
        out = self.dense(self.flatten(convolution, training=training), training=training)

        out = tf.keras.activations.relu(out, alpha=0.0, max_value=None, threshold=0)
        out = self.gru(residual, out)
        out = self.dropout(out, training=training)

        return out  # (batch_size, input_seq_len, d_model)
