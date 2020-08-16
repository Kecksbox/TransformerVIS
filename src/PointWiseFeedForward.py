import tensorflow as tf


def PointWiseFeedForward(d_model, dff, num_layers=2):
    assert num_layers > 1
    layers = [None] * num_layers
    layers[-1] = tf.keras.layers.Dense(d_model)
    for i in range(num_layers - 1):
        layers[i] = tf.keras.layers.Dense(dff, activation='relu')
    return tf.keras.Sequential(layers)
