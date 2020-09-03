import json

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA

from src.EncodingAttentionAutoEncoder.EncodingAttentionAutoEncoder import EncodingAttentionAutoEncoder
from src.InputPipeline import InputPipeline
from src.SelfAttentionAutoEncoder.SelfAttentionAutoEncoder import SelfAttentionAutoEncoder
from src.Utilities.ConvolutionBuilder import compute_convolutions
from src.Utilities.GameOfLife2 import createTestSet, show
from src.Utilities.TokenBuilder import createFullToken
from src.Model import Model

train_examples = createTestSet()

"""
d_model = 80
model = Model(
    shape=(None, 5, 5, 1, 1),
    d_model=d_model,
    num_convolutions=3,
    num_heads=4,
    encoder_specs=[
        #  (dff, d_tar)
        (112, 16),
        (112, 6),
    ],
    num_layers_decoder=4,
    dff_decoder=112,
    max_length=22,
    BATCH_SIZE=64,
    dropout_rate=0.0,
    PAD_TOKEN=-10,
)
"""
SOS = createFullToken((5, 5, 1, 1), -1)
EOS = createFullToken((5, 5, 1, 1), -2)
input_pipeline = InputPipeline(
    BUFFER_SIZE=2000,
    BATCH_SIZE=2000,
    shape=(None, 5, 5, 1, 1),
    max_length=22,
    SOS=SOS,
    EOS=EOS,
    PAD_TOKEN=-10,
)
"""
orignalModel = Model(
    shape=(None, 5, 5, 1, 1),
    d_model=80,
    num_convolutions=2,
    num_heads=2,
    encoder_specs=[
        #  (dff, d_tar)
        (224, 32),
        (224, 16),
        (224, 2),
    ],
    num_layers_decoder=2,
    dff_decoder=112,
    max_length=22,
    BATCH_SIZE=1000,
    dropout_rate=0.0,
    PAD_TOKEN=-10,
)
"""
selfAttentionAutoEncoder = SelfAttentionAutoEncoder(
    voxel_shape=(5, 5, 1, 1),
    d_model=81,
    d_parameter=112,
    seq_convolution=compute_convolutions(num_convolutions=2, voxel_shape=(5, 5, 1, 1), d_model=81,
                                         convolution_scaling=1),
    static_convolution=compute_convolutions(num_convolutions=2, voxel_shape=(5, 5, 2, 1), d_model=112,
                                            convolution_scaling=1),
    num_attention_layers=3, attention_dff=224,
    num_decoder_layers=3, decoder_dff=224,
    max_length=22, SOS=-1, EOS=-2, PAD_TOKEN=-10,
    rate=0.000)
encodingAutoEncoder = EncodingAttentionAutoEncoder(
    voxel_shape=(5, 5, 1, 1),
    d_model=80,
    seq_convolution=compute_convolutions(num_convolutions=3, voxel_shape=(5, 5, 1, 1), d_model=80,
                                         convolution_scaling=2),
    # ([index: (dff, d_tar)])
    encoder_specs=[
        (112, 32),
        (112, 16),
        (112, 16),
        (112, 2),
    ],
    num_heads=4,
    num_layers_decoder=2, dff_decoder=224,
    max_length=22, SOS=-1, EOS=-2, PAD_TOKEN=-10,
    rate=0.000)

set = input_pipeline.process(train_examples)
# encodingAutoEncoder.train(set, 80000)
# selfAttentionAutoEncoder.train(set, 80000)
# orignalModel.train(set, 80000)

data = {}
data[0] = []
test_latent = []
i = 0
max = 40
for (parameters, run) in train_examples:
    i += 1
    if i == max:
        break
    encodingReconstruction, latent, encoding_attention_weights = encodingAutoEncoder.evaluate(
        run
    )
    selfAttentionReconstruction, self_attention_weights = selfAttentionAutoEncoder.evaluate(
        run, parameters
    )

    """
    reconstruction, latent, attention = orignalModel.evaluate(run)
    self_attention_weights = {}
    encoding_attention_weights = {}
    j = 0
    for key in attention:
        j += 1
        if j % 2 == 0:
            self_attention_weights[key] = attention[key]
        else:
            encoding_attention_weights[key] = attention[key]
    """
    data[0].append([selfAttentionReconstruction[1:-1], latent, [self_attention_weights, encoding_attention_weights]])
    test_latent.append(latent)
    # show(e[:1])
    # show(test[:1])
    # show(e[1:2])
    # show(test[1:2])
    #show(run)
    #show(reconstruction[1:-1])
    #show(encodingReconstruction)
    #show(selfAttentionReconstruction[1:-1])


def pca(inputs, n_components):  # inputs (batch, time_steps, d_latent)
    # turn inputs into a matrix
    features = tf.concat(tf.unstack(inputs), axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(features)
    features = pca.transform(features)

    return tf.reshape(features, (inputs.shape[0], inputs.shape[1], n_components))


def normalize(data):
    # creates a copy of data
    X = tf.identity(data)
    # calculates the mean
    X -= tf.reduce_mean(data, axis=0)
    return X


# data = pca_fn(tf.transpose(X))

# could have non uniform timesteps
comb = test_latent
"""
# test = tf.concat(comb, axis=0)
max_dim = -1
for e in comb:
    if e.shape[1] > max_dim:
        max_dim = e.shape[1]
# now clone last eleemnt for all that have to be padded
comb_length = comb.__len__()
adjustments = [0] * comb_length
for i in range(comb_length):
    e = comb[i]
    diff = max_dim - e.shape[1]
    adjustments[i] = diff
    if diff > 0:
        adjustment = [1] * (e.shape[1])
        adjustment[-1] += diff
        comb[i] = tf.repeat(e, adjustment, axis=1)
comb = tf.concat(comb, axis=0)
comb = pca(comb, 1)
# remove again
comb = tf.unstack(comb, axis=0)
for i in range(comb_length):
    e = comb[i]
    if adjustments[i] > 0:
        adjustment = [1] * (e.shape[0] - adjustments[i]) + [0] * adjustments[i]
        comb[i] = tf.repeat(e, adjustment, axis=0)
"""
X = comb


def draw(run):
    plt.plot(range(run.shape[0]), run[:, 0])

for run in X:
    draw(tf.squeeze(run, axis=0))
plt.ylabel('some numbers')
plt.show()

for i in range(X.__len__()):
    data[0][i][1] = tf.squeeze(X[i], axis=0).numpy().tolist()

for i in range(data[0].__len__()):
    element = data[0][i]
    element[0] = element[0].numpy().tolist()
    for attentionType in element[2]:
        for key in attentionType:
            attentionType[key] = attentionType[key].numpy().tolist()

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)
