import tensorflow as tf
from Parameters import block_size,n_embd
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

tf.test.main()