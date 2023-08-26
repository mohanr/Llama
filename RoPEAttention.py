import tensorflow as tf
from Parameters import n_embd, n_heads, block_size, batch_size
from RotaryPositionalEmbeddings import RotaryPositionalEmbeddings


class RoPEAttention(tf.keras.Model):

    def __init__(self):
        super(RoPEAttention, self).__init__()
        self.w_q = tf.keras.layers.Dense(n_embd, input_shape=(n_embd,), use_bias=False)
        self.w_k = tf.keras.layers.Dense(n_embd, input_shape=(n_embd,), use_bias=False)
        self.w_v = tf.keras.layers.Dense(n_embd, input_shape=(n_embd,), use_bias=False)
        self.multihead = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=n_embd//n_heads, dropout=0.1)
        print(n_embd//n_heads)
        rpe = RotaryPositionalEmbeddings()
        self.R = rpe.rotary_matrix(block_size,n_embd)

    def call(self, x, return_attn_weights=False):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        print(f'Shape of R,q {tf.shape(self.R)}{tf.shape(q)}{tf.shape(tf.transpose(q, perm=[1, 0, 2]))}')
        q_t = tf.transpose(q, perm=[1, 0, 2])  # Transpose q from (batch_size, seq_length, q_dim) to (batch_size, q_dim, seq_length)
        q_out = tf.transpose(tf.matmul(q_t, self.R),perm=[1, 0, 2])  # Transpose back to (batch_size, seq_length, r_dim)
        k_t = tf.transpose(k, perm=[1, 0, 2])  # Transpose q from (batch_size, seq_length, q_dim) to (batch_size, q_dim, seq_length)
        k_out = tf.transpose(tf.matmul(k_t, self.R),perm=[1, 0, 2])  # Transpose back to (batch_size, seq_length, r_dim)
        v_t = tf.transpose(v, perm=[1, 0, 2])  # Transpose q from (batch_size, seq_length, q_dim) to (batch_size, q_dim, seq_length)
        v_out = tf.transpose(tf.matmul(v_t, self.R),perm=[1, 0, 2])  # Transpose back to (batch_size, seq_length, r_dim)
        print(f'Shapes of key,query and value are {tf.shape(k_out)}{tf.shape(q_out)}{tf.shape(v_out)}')
        activations, attn_weights = self.multihead(query=q_out,value=v_out,key=k_out, return_attention_scores=True)

        if return_attn_weights:
            return activations, attn_weights
        return activations

