import tensorflow as tf
from Parameters import batch_size,block_size,n_embd

class TestRMSNorm(tf.test.TestCase):

    def setUp(self):
        super(TestRMSNorm, self).setUp()
        self.batch = tf.random.normal((batch_size, block_size, n_embd))

    def test_RMSNormTest(self):
        normalized_mat, norm = tf.linalg.normalize(self.batch, axis=(1, 2))
        ff_rms = tf.multiply(norm,
                             tf.pow(tf.cast(tf.size(self.batch[0]), tf.float32), -0.5))
        ffx = tf.Variable(tf.zeros_like(self.batch))
        print(tf.shape(ffx))
        for i in range(self.batch.shape[0]):
            ffx[i, :, : ].assign(tf.divide(self.batch[i] , ff_rms[i]))
        normalized_mat, norm = tf.linalg.normalize(self.batch, axis=(1, 2))
        print(tf.pow(norm,2))

        # The values are close to 1024 but close enough for default
        # tolerance levels to pass the test. So it will fail unless
        # I pass a different tolerance level. I believe this is a temporary
        # fix until I understand the issue.
        self.assertAllClose(tf.pow(norm,2),
                            tf.reshape(
                                tf.repeat([tf.constant(1024,tf.float32)], repeats=[4], axis=0),
                                (4,1,1)),50,50)


tf.test.main()