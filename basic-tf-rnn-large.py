"""
Minimal character-level Vanilla RNN model.
Author: Mario Meissner
"""

import numpy as np
import tensorflow as tf
# data I/O
data = open('samples/star_wars.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(('data has %d characters, %d unique.' % (data_size, vocab_size)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 500  # size of hidden layer of neurons
seq_length = 50  # number of steps to unroll the RNN for
batch_size = 10 # Number of inputs to evaluate simultaniously
learning_rate = 2e-1

# create the TF graph
with tf.device('/device:GPU:0'):

    Wxh = tf.Variable(np.random.randn(hidden_size, batch_size) * 0.01, dtype=tf.float32)  # input to hidden
    Whh = tf.Variable(np.random.randn(hidden_size, hidden_size) * 0.01, dtype=tf.float32)  # hidden to hidden
    Why = tf.Variable(np.random.randn(vocab_size, hidden_size) * 0.01, dtype=tf.float32)  # hidden to output
    bh = tf.Variable(tf.zeros((1, hidden_size)), dtype=tf.float32)  # hidden bias
    by = tf.Variable(tf.zeros((1, vocab_size)), dtype=tf.float32)  # output bias
    x_batches = tf.placeholder(tf.float32, (batch_size, seq_length))  # our input
    y_batches = tf.placeholder(tf.int32, (batch_size, seq_length))  # our expected output label
    init_h = tf.placeholder(tf.float32, [batch_size, hidden_size])

    # Unpack the columns (each will be one batch)
    x_batches_list = tf.unstack(x_batches, axis=1)
    y_batches_list = tf.unstack(y_batches, axis=1)

    # Unroll graph for seq_length timesstamps
    h_sequence = list()
    h_sequence.append(init_h)
    current_h = init_h
    for x_batch in x_batches_list:
        # Add the second dimension back (x_batch will be a column)
        x_batch = tf.expand_dims(x_batch, 1)
        next_h = tf.tanh(tf.matmul(Whh, current_h) +
            tf.matmul(Wxh, x_batch) + bh)
        h_sequence.append(next_h)
        current_h = next_h

    logits = [tf.matmul(Why, h) + by for h in h_sequence]
    pred_sequence = tf.nn.softmax(logits)
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_batches)]
    mean_loss = tf.reduce_mean(losses)
    init = tf.global_variables_initializer()
    train_sequence = tf.train.AdagradOptimizer(0.3).minimize(mean_loss)


def sample(h, seed_ix, n):
    """
    sample a sequence of n integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    #  TODO: Rewrite this using tf
    return


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config = config) as sess:
    sess.run(init)
    loss_list = []

    n, p = 0, 0
    _current_state = np.random.randn(hidden_size, 1)
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size, 1))  # reset RNN memory
            p = 0  # go from start of data
        input_chars_ix = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        target_chars_ix = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

        # we need to create the input and target arrays of shape vocabsize x seq_length
        input_oneshots = np.zeros((vocab_size, seq_length))
        target_oneshots = np.zeros((vocab_size, seq_length))
        for t in range(seq_length):
            input_oneshots[input_chars_ix[t]][t] = 1  # set the char position to 1
            target_oneshots[target_chars_ix[t]][t] = 1  # set the char position to 1

        # sample from the model now and then
        if n % 500 == 0:
            # seed_ix = inputs[0]
            # sample_ix = sample(hprev, seed_ix, 400)
            print("I am processing")
            # txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            # print(('----\n %s \n----' % (txt,)))


        # forward seq_length characters through the net and fetch gradient
        _mean_loss, _train_sequence, _current_state, _pred_sequence = sess.run(
            [mean_loss, train_sequence, current_h, pred_sequence],
            feed_dict={
                x_batches: input_oneshots,
                y_batches: target_oneshots,
                init_h: _current_state
            })
        if n % 1000 == 0: print(('iter %d, loss: %f' % (n, _mean_loss)))  # print progress

        p += seq_length  # move data pointer
        n += 1  # iteration counter



