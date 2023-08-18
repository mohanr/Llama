import tensorflow as tf
from keras.layers import Embedding
import tensorflow_probability as tfp
from Parameters import n_embd,block_size
from RMSNorm import RMSNorm


class InitialModel(tf.keras.Model):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size,n_embd)
        self.rms = RMSNorm([block_size, n_embd])

        self.net = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(n_embd, input_shape=(None,n_embd), activation=None, use_bias=False),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(vocab_size, input_shape=(n_embd,), activation=None, use_bias=False),
            ]
        )

    def call(self,idx,targets=None):
        x = self.token_embedding_table(idx)
        x = self.rms(x)
        logits = self.net(x)

        # print(f'Shape of logits {tf.shape(logits)} , targets {tf.shape(targets)}')
        if targets is None:
            loss = None
        else:
            bce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = bce(targets,logits)
        return logits, loss

    def generate(self,idx,max_new_tokens):
        i = tf.constant(0)
        c = lambda i, d: tf.less(i, max_new_tokens)

        def b(i, idx):
            # print(tf.shape(idx))
            idx_cond = idx[-block_size:]
            logits,loss = self(idx_cond)
            logits = logits[-1:, :,:]
            probs = tf.nn.softmax(logits)
            idx_next = tfp.distributions.Multinomial(total_count=1,probs=probs)
            sample = idx_next.sample(1)
            idx = tf.concat([idx,
                        tf.cast(tf.where(
                            tf.squeeze(sample)),tf.int64)
                      ],0)
            return tf.add(i, 1), idx

        _, idx = tf.while_loop(c, b, loop_vars=[i, idx])
        # print(f'idx in generate is {idx}')
        return idx
