import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy
from src.Utilities.GameOfLife import createTestSet, show

from src.Model import Model

"""
def create2x2x2x2Vector(value):
    return [
        [
            [
                [value, value],
                [value, value],
            ],
            [
                [value, value],
                [value, value],
            ],
        ],
        [
            [
                [value, value],
                [value, value],
            ],
            [
                [value, value],
                [value, value],
            ],
        ],
    ]  # (2, 2, 2, 2) 3d volume with vectors in each voxel


dummy_data = [
    np.array([
        create2x2x2x2Vector(1),
    ]),
    np.array([
        create2x2x2x2Vector(2),
        create2x2x2x2Vector(3),
        create2x2x2x2Vector(4),
    ]),
    np.array([
        create2x2x2x2Vector(3),
        create2x2x2x2Vector(4),
        create2x2x2x2Vector(5),
    ]),
    np.array([
        create2x2x2x2Vector(2),
        create2x2x2x2Vector(9),
    ])
]


def gather_dummy_data():
    for data in dummy_data:
        yield data


train_examples = tf.data.Dataset.from_generator(gather_dummy_data, output_types=tf.float32,
                                                output_shapes=(None, 2, 2, 2, 2))
"""

train_examples = createTestSet(20)

d_model = 200
model = Model(
    shape=(None, 10, 10, 1, 1),
    d_model=d_model,
    num_convolutions=2,
    num_heads=4,
    encoder_specs=[
        #  (dff, d_tar)
        (300, 240),
        (240, 160),
        (200, 160),
        (200, 112),
        (400, 60),
    ],
    num_layers_decoder=4,
    dff_decoder=240,
    max_length=12,
    BATCH_SIZE=10,
    dropout_rate=0.0,
    PAD_TOKEN=-10,
)

# set = model.train(train_examples, 10000)

test_results = []
test_latent = []
for e in train_examples:
    result, latent, attention_weights = model.evaluate(tf.constant(
        e
    , tf.float32))
    test_results.append(result)
    test_latent.append(latent)
    test = result[1:-1]
    show(e[:1])
    show(test[:1])
    show(e[1:2])
    show(test[1:2])
    show(e)
    show(test)

def pca(inputs, n_components): # inputs (batch, time_steps, d_latent)
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
 X -=tf.reduce_mean(data, axis=0)
 return X

# data = pca_fn(tf.transpose(X))

# could have non uniform timesteps
comb = test_latent
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

X = comb

def draw(run):
    plt.plot(range(run.shape[0]), run[:, 0])


for run in X:
    draw(run)
plt.ylabel('some numbers')
plt.show()

