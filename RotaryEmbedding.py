from matplotlib import pyplot as plt
import tensorflow as tf
from Parameters import n_embd
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
    plt.show()


if __name__ == "__main__":
    rotaryPositionalEmbeddingsTest()