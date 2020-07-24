import tensorflow as tf

from src.InputPipeline import InputPipeline
from src.Trainer import Trainer
from src.Transformer import Transformer


def createFullToken(voxel_shape, TOKEN):
    return tf.fill(voxel_shape, TOKEN)


class Model:

    def __init__(self,
                 shape, max_length,
                 d_model, d_latent, dff, num_layers, num_heads,
                 SOS_TOKEN=-1, EOS_TOKEN=-2, PAD_TOKEN=0,
                 BUFFER_SIZE=20000, BATCH_SIZE=64,
                 dropout_rate=0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9, warmup_steps=4000,
                 ):
        voxel_shape = shape[1:]
        SOS = createFullToken(voxel_shape, SOS_TOKEN)
        EOS = createFullToken(voxel_shape, EOS_TOKEN)
        self.input_pipeline = InputPipeline(
            BUFFER_SIZE=BUFFER_SIZE,
            BATCH_SIZE=BATCH_SIZE,
            shape=shape,
            max_length=max_length,
            SOS=SOS,
            EOS=EOS,
            PAD_TOKEN=PAD_TOKEN,
        )
        self.trainer = Trainer(
            transformer=Transformer(
                voxel_shape=voxel_shape,
                d_model=d_model,
                d_latent=d_latent,
                num_layers=num_layers,
                num_heads=num_heads,
                dff=dff,
                max_length=max_length,
                SOS=SOS,
                EOS=EOS,
                rate=dropout_rate
            ),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            warmup_steps=warmup_steps,
        )

    def train(self, train_dataset, epochs):
        self.trainer.train(
            train_dataset=self.input_pipeline.process(train_dataset),
            epochs=epochs
        )

    def evaluate(self, inp):
        return self.trainer.transformer.evaluate(inp)
