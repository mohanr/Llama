import tensorflow as tf
from Parameters import batch_size,block_size,n_embd
import numpy as np

class RMSNorm(tf.keras.Model):

    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.scale = tf.Variable(initial_value=np.ones([block_size,n_embd]), trainable=True,dtype=tf.float32)

    def call(self, x):
        normalized_mat, norm = tf.linalg.normalize(x, axis=(1, 2))
        # print(f'Normalize {norm}')
        print(f'Size {tf.size(x[0])}')
        ff_rms = tf.multiply(norm ,
                             tf.pow(tf.cast(tf.size(x[0]),tf.float32),-0.5))
        print(f'Shape of ff_rms {tf.shape(ff_rms)}')
        raw = tf.divide(x , ff_rms )
        print(f' Shape of raw {tf.shape(raw)} {self.scale[:tf.shape(x)[1], :]}')
        return tf.multiply(self.scale[:tf.shape(x)[1], :] , raw)

