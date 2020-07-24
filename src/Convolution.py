import tensorflow as tf

class Convolution(tf.keras.layers.Layer):
    def __init__(self, input_shape, num_layers, d_model, rate=0.1):
        super(Convolution, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.conv_layers = [
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(filters=10, kernel_size=1, strides=1, activation='relu', input_shape=input_shape)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(filters=2, kernel_size=1, strides=1, activation='relu'))
        ]

        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

        self.dense = tf.keras.layers.Dense(d_model, activation=None)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        residual = self.dense(self.flatten(x, training=training))

        for conv_layer in self.conv_layers:
            x = conv_layer(x, training=training)
        x = self.flatten(x, training=training)
        x = self.dense(x, training=training)
        x = self.dropout(x, training=training)

        return residual + x  # (batch_size, input_seq_len, d_model)
