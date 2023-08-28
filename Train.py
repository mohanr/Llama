import datetime

import tensorflow as tf

from Dataset import decode
from Dataset import draw_random_sample_batches
from Parameters import block_size, learning_rate

# model = InitialModel(vocab_size)
from RoPEModel import RoPEModel

def write_summary(loss, epoch):

    with train_summary_writer.as_default():

        tf.summary.scalar('Ilama loss', loss, step=epoch)

model = RoPEModel()
x, y = draw_random_sample_batches(block_size)
logits, loss = model(x, y)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

logdir = "/Users/anu/PycharmProjects/TensorFlow2/logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(logdir)

epochs = 1
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    for step in range(10000):
        with tf.GradientTape() as tape:
            x,y = draw_random_sample_batches(block_size)
            logits,loss = model(x,y)
        grads = tape.gradient(loss, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss at step %d: %.4f"
                % (step, float(loss))
            )
            write_summary(loss=float(loss),epoch=step)
            print("Seen so far: %s samples" % ((step + 1)))
