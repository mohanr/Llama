import tensorflow as tf
from RotaryEmbeddings import RotaryEmbeddings

class TestRMSNorm(tf.test.TestCase):

    def setUp(self):
        super(TestRMSNorm, self).setUp()
        self.x = tf.constant([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=tf.float32)

    def test_rotary(self):

        self.x = self.x[:, None, None, :]
        rpe = RotaryEmbeddings(4)
        self.assertAllClose(rpe(self.x),tf.constant([[[[  1.,2.,3.,4.]]],
                                                [[[ -2.8876166,   4.9297514,   6.6076975,   7.0496492]]],
                                                [[[-11.0967045,   7.7984138,   2.6197603,  10.1579895]]]]))

tf.test.main()