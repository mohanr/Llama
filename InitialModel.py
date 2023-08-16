import tensorflow as tf
from keras.layers import Embedding

from Parameters import n_embd


class InitialModel(tf.keras.Model):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size,n_embd)

        self.net = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(n_embd, input_shape=(None,n_embd), activation=None, use_bias=False),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(vocab_size, input_shape=(n_embd,), activation=None, use_bias=False),
            ]
        )

    def call(self,idx,targets=None):
        x = self.token_embedding_table(idx)
        logits = self.net(x)

        # print(f'Shape of logits {tf.shape(logits)} , targets {tf.shape(targets)}')
        if targets is None:
            loss = None
        else:
            bce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = bce(targets,logits)
        return logits, loss
