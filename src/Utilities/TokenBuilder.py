import tensorflow as tf


def createFullToken(voxel_shape, TOKEN):
    return tf.fill(voxel_shape, TOKEN)
