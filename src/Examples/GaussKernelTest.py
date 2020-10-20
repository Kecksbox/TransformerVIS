import tensorflow as tf

tf.random.set_seed(1)
# tf.config.set_visible_devices([], 'GPU')

import numpy as np
from src.EncodingAttentionAutoEncoder.EncodingAttentionAutoEncoder import EncodingAttentionAutoEncoder
from src.Utilities.InputPipeline import InputPipeline
from src.Examples.Games.GameOfLife2 import show
from src.Utilities.TokenBuilder import createFullToken

size = 512
SOS = createFullToken((size, size, 1, 1), -1)
EOS = createFullToken((size, size, 1, 1), -2)
paramter_shape = (size, size, 2, 1)
inp_shape = (None, size, size, 1, 1)

def createTestSet_internal():
    np_path = "./Datasets/ensemble1_np/"
    for i in range(1, 4):
        yield (
            tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant(0, shape=(1,)), axis=-1), axis=-1), axis=-1),
            tf.expand_dims(tf.expand_dims(np.load(np_path + "Sim_{}.npy".format(i)), axis=-1), axis=-1),
        )

def createTestSet():
    return tf.data.Dataset.from_generator(
        createTestSet_internal,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([1, 1, 1, 1]), tf.TensorShape([None, size, size, 1, 1]))
    )

train_examples = createTestSet()

input_pipeline = InputPipeline(
    BUFFER_SIZE=3,
    BATCH_SIZE=1,
    max_length=22,
    SOS=SOS,
    EOS=EOS,
    PAD_TOKEN=-10,
)
set = input_pipeline.process(train_examples, paramter_shape=paramter_shape, inp_shape=inp_shape)

encodingAutoEncoder = EncodingAttentionAutoEncoder(
    voxel_shape=inp_shape[1:],
    d_model=112,
    seq_convolution=[dict(filters=6, kernel_size=[32, 32, 1], strides=[16, 16, 1]), dict(filters=12, kernel_size=[3, 3, 1], strides=[2, 2, 1])],
    # ([index: (dff, d_tar)])
    encoder_specs=[
        (112, 112),
        (112, 60),
        (112, 32),
        (112, 2),
    ],
    num_attention_layers=2, att_dff=112,
    num_layers_decoder=2, dff_decoder=112,
    max_length=102, SOS=-1, EOS=-2, PAD_TOKEN=-10,
    rate=0.000
)

encodingAutoEncoder.train(set, 80000)

i = 0
max = 40
test_latent = []
runs = []
for (parameters, run) in train_examples:
    i += 1
    if i == max:
        break
    encodingReconstruction, latent, encoding_attention_weights = encodingAutoEncoder.evaluate(
        run
    )
    test_latent.append(latent)
    runs.append(run)

show(runs, test_latent, save=True)
