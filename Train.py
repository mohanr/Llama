import datetime

import tensorflow as tf

from Dataset import draw_random_sample_batches, vocab_size
from InitialModel import InitialModel
from Parameters import block_size, learning_rate

m = InitialModel(vocab_size)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

logdir = "/Users/anu/PycharmProjects/TensorFlow2/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
epochs = 1
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    for step in range(10000):
        with tf.GradientTape() as tape:
            x,y = draw_random_sample_batches(block_size)
            logits,loss = m(x,y)

        grads = tape.gradient(loss, m.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, m.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss at step %d: %.4f"
                % (step, float(loss))
            )
            print("Seen so far: %s samples" % ((step + 1)))
