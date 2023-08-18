import numpy as np
import tensorflow as tf


class RMSNorm(tf.keras.Model):

    def __init__(self, layer_shape):
        super(RMSNorm, self).__init__()
        self.scale = tf.Variable(initial_value=np.ones(layer_shape), trainable=True,dtype=tf.float32)

    def call(self, x):
        normalized_mat, norm = tf.linalg.normalize(x, axis=(1, 2))
        # print(f'Normalize {norm}')
        rms = tf.multiply(norm ,
                             tf.pow(tf.cast(tf.size(x[0]),tf.float32),-0.5))
        r = tf.divide(x , rms )
        return tf.multiply(self.scale[:tf.shape(x)[1], :] , r)

