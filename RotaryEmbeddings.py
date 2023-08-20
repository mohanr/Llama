import tensorflow as tf

# Directly ported to TensorFlow based on https://nn.labml.ai/transformers/rope/index.html
# The math has to be fully understood.
class RotaryEmbeddings(tf.keras.Model):

    def __init__(self, d, base = 10000):
        super(RotaryEmbeddings, self).__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def build_cache(self, x):
            if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
                return
            seq_len = x.shape[0]
            theta = 1. / (self.base ** (tf.range(0, self.d, 2) / self.d))
            seq_idx = tf.range(seq_len)
            idx_theta = tf.einsum('n,d->nd',
                                  tf.cast(seq_idx,tf.float32),
                                  tf.cast(theta,tf.float32))
            idx_theta2 = tf.concat([idx_theta, idx_theta], axis=1)
            self.cos_cached = tf.cos(idx_theta2)[:, None, None, :]
            self.sin_cached = tf.sin(idx_theta2)[:, None, None, :]

    def neg_half(self, x):
            d_2 = tf.cast(tf.divide(self.d , 2), tf.int32 )
            return tf.concat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], axis=-1)

    def call(self, x):
            self.build_cache(x)
            x_rope, x_pass = x[..., :self.d], x[..., self.d:]
            neg_half_x = self.neg_half(x_rope)
            x_rope = tf.add(tf.multiply(x_rope , self.cos_cached[:x.shape[0]])
                     ,
                     tf.multiply(neg_half_x , self.sin_cached[:x.shape[0]]))
            return tf.concat((x_rope, x_pass), axis=-1)