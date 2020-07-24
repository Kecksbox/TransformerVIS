import numpy as np
import tensorflow as tf

from src.Model import Model


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
    ])
]


def gather_dummy_data():
    for i in range(2):
        yield dummy_data[i]


train_examples = tf.data.Dataset.from_generator(gather_dummy_data, output_types=tf.float32,
                                                output_shapes=(None, 2, 2, 2, 2))
for e in train_examples:
    print(e)

"""
The dummy set is created
"""

# this is all that has to be provided -------------------------------- end

max_length = 5

input_shape = (2, 2, 2, 2)
num_layers = 4
d_model = 5
d_latent = 5
dff = 20
num_heads = 5
dropout_rate = 0.0

model = Model(
    shape=(None, 2, 2, 2, 2),
    d_model=5,
    d_latent=5,
    dff=20,
    num_layers=4,
    num_heads=5,
    max_length=5,
)

# model.train(train_examples, 1000)

result, latent, attention_weights = model.evaluate(tf.constant(
    [
        create2x2x2x2Vector(1),
    ]
    , tf.float32))

print(result)
print(latent)
print(attention_weights)
