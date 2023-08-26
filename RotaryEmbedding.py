from matplotlib import pyplot as plt
import tensorflow as tf
from Parameters import n_embd, batch_size, block_size
from RoPEAttention import RoPEAttention
from RotaryPositionalEmbeddings import RotaryPositionalEmbeddings


def rotaryPositionalEmbeddingsTest():
    K = 3
    rotary_emb = RotaryPositionalEmbeddings()
    R = rotary_emb.rotary_matrix(tf.pow(K,2),n_embd)
    fig, ax = plt.subplots(K, K, figsize=(K * 3, K * 4))

    for i in range(K):
        for j in range(K):
            ax[i, j].imshow(R[i * K + j, :, :])
            ax[i, j].set_title(f'rotation at {i * K + j}')
    # plt.show()
    plt.savefig("rotraryembeddings.png")

def ropeAttentionTest():
    layer = RoPEAttention()
    batch = tf.random.uniform((batch_size, block_size, n_embd))
    output, attn_weights = layer(batch, return_attn_weights=True)
    print(f'Shape of attention weights is {tf.shape(attn_weights)}')
    weight_names = ['query', 'keys', 'values', 'proj']
    for name, out in zip(weight_names, layer.get_weights()):
        print(name, out.shape)
    plt.imshow(attn_weights[0][0], interpolation='nearest')
    plt.colorbar()
    data = tf.random.normal(( 12 , 12 ))
    # plt.imshow( data )
    plt.show()

if __name__ == "__main__":
    # rotaryPositionalEmbeddingsTest()
    ropeAttentionTest()