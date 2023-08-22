import tensorflow as tf
import numpy as np

class RotaryPositionalEmbeddings(tf.keras.Model):

    def __init__(self):
        super(RotaryPositionalEmbeddings, self).__init__()

    def rotary_matrix( self,block_size, embedding_dim):
        R = tf.Variable(tf.zeros((block_size, embedding_dim, embedding_dim)))
        i = tf.constant(0)
        p_i = tf.constant(0)
        neg_2 = tf.constant(-2)
        emb = lambda i, d: tf.less(i, int(tf.divide(embedding_dim , 2) - 1))
        p = lambda p_i, d: tf.less(p_i, block_size )
        print(int(tf.divide(embedding_dim , 2)))
        def position(p_i, p_idx):
            def embedding(i, idx):
                theta = tf.pow(10000. , tf.divide(tf.multiply(neg_2 , tf.subtract(i , 1)) , embedding_dim))
                m_theta = tf.multiply(tf.cast(p_i,tf.float32) , tf.cast(theta,tf.float32))
                R[p_i, tf.multiply(2, i),tf.multiply(2, i)].assign(tf.cos(m_theta))
                # print(i, p_i, tf.multiply(2, i), tf.multiply(2, tf.add(i , 1)))
                R[p_i, tf.multiply(2, i), tf.multiply(2, tf.add(i , 1))].assign(- tf.sin(m_theta))
                R[p_i, tf.multiply(2, tf.add(i , 1)), tf.multiply(2, i)].assign(tf.sin(m_theta))
                R[p_i, tf.multiply(2, tf.add(i , 1)), tf.multiply(2, tf.add(i , 1))].assign(tf.cos(m_theta))

                return tf.add(i, 1), idx

            _, idx = tf.while_loop(emb, embedding, loop_vars=[i, embedding_dim])
            return tf.add(p_i, 1), p_idx


        _, idx = tf.while_loop(p, position, loop_vars=[p_i, block_size])
        return R
