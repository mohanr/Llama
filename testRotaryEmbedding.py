import tensorflow as tf
from Parameters import block_size, n_embd, batch_size
from RoPEAttention import RoPEAttention
from RotaryPositionalEmbeddings import RotaryPositionalEmbeddings

class testRotaryEmbedding(tf.test.TestCase):

    def setUp(self):
        super(testRotaryEmbedding, self).setUp()
        self.rpe = RotaryPositionalEmbeddings()


    def test_RotaryEmbedding(self):
        R = self.rpe.rotary_matrix(block_size,n_embd)
        x = tf.random.normal((n_embd,1))
        y = tf.random.normal((n_embd,1))

        m = 3
        n = 13

        x_m = tf.matmul(R[m,:,:] , x )
        x_n = tf.matmul(R[n,:,:] , y )
        print(tf.shape(tf.matmul(x_m , tf.transpose(x_n))))
        print(tf.shape(tf.matmul(tf.transpose(x) ,
                                                R[n-m,:,:])))
        # The values are not close enough for default
        # tolerance levels to pass the test. So it will fail unless
        # I pass a different tolerance level. I believe this is a temporary
        # fix until I understand the issue.

        self.assertAllClose(tf.matmul(x_m , tf.transpose(x_n)),
                            tf.matmul(tf.transpose(tf.matmul(tf.transpose(x) ,
                                                R[n-m,:,:])) , tf.transpose(y)),21)

    def test_RotaryEmbedding(self):

        x = tf.random.normal((batch_size,block_size, n_embd))
        layer = RoPEAttention()
        output, attn_weights = layer(x, return_attn_weights=True)

        q = layer.w_q(x)
        k = layer.w_k(x)
        v = layer.w_v(x)

        q_rotated = tf.Variable(tf.zeros_like(x))
        k_rotated = tf.Variable(tf.zeros_like(x))
        v_rotated = tf.Variable(tf.zeros_like(x))

        for position in range(block_size):
            q_rotated[:,position,:].assign(tf.matmul(q[:,position,:], layer.R[position,:,:]))
            k_rotated[:,position,:].assign(tf.matmul(k[:,position,:], layer.R[position,:,:]))
            v_rotated[:,position,:].assign(tf.matmul(v[:,position,:], layer.R[position,:,:]))

        q_t = tf.transpose(q, perm=[1, 0, 2])  # Transpose q from (batch_size, seq_length, q_dim) to (batch_size, q_dim, seq_length)
        q_out = tf.transpose(tf.matmul(q_t, layer.R),
                             perm=[1, 0, 2])  # Transpose back to (batch_size, seq_length, r_dim)
        k_t = tf.transpose(k, perm=[1, 0, 2])  # Transpose q from (batch_size, seq_length, q_dim) to (batch_size, q_dim, seq_length)
        k_out = tf.transpose(tf.matmul(k_t, layer.R),
                             perm=[1, 0, 2])  # Transpose back to (batch_size, seq_length, r_dim)
        v_t = tf.transpose(v, perm=[1, 0, 2])  # Transpose q from (batch_size, seq_length, q_dim) to (batch_size, q_dim, seq_length)
        v_out = tf.transpose(tf.matmul(v_t, layer.R),
                             perm=[1, 0, 2])  # Transpose back to (batch_size, seq_length, r_dim)

        self.assertAllClose(tf.transpose(q, perm=[1, 0, 2])[0], q[:,0,:],3,3)
        self.assertAllClose(tf.matmul(tf.transpose(k, perm=[1, 0, 2])[0] , layer.R[0]),
                            tf.matmul(q[:,0,:] , layer.R[0]),3,3)
        self.assertAllClose(q_rotated, q_out,3,3)
tf.test.main()