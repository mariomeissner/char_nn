"""
Minimal character-level Vanilla RNN model.
Author: Mario Meissner
"""

import numpy as np
import tensorflow as tf

# data I/O
data = open('samples/arxiv_abstracts.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(('data has %d characters, %d unique.' % (data_size, vocab_size)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 50  # number of steps to unroll the RNN for
learning_rate = 5e-1

# create the TF graph
with tf.device('/device:GPU:0'):

    Wxh = tf.Variable(np.random.randn(hidden_size, vocab_size) * 0.01, dtype=tf.float32)  # input to hidden
    Whh = tf.Variable(np.random.randn(hidden_size, hidden_size) * 0.01, dtype=tf.float32)  # hidden to hidden
    Why = tf.Variable(np.random.randn(vocab_size, hidden_size) * 0.01, dtype=tf.float32)  # hidden to output
    bh = tf.Variable(tf.zeros((hidden_size, 1)), dtype=tf.float32)  # hidden bias
    by = tf.Variable(tf.zeros((vocab_size, 1)), dtype=tf.float32)  # output bias
    # Each row will be an entry
    x_series = tf.placeholder(tf.float32, (seq_length, vocab_size))  # our input
    y_series = tf.placeholder(tf.int32, (seq_length, vocab_size))  # our expected output label
    cell_state = tf.placeholder(tf.float32, [hidden_size, 1])
    hidden_state = tf.placeholder(tf.float32, [hidden_size, 1])
    init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state);
    single_input = tf.placeholder(tf.float32, (vocab_size, 1))
    # Unpack the rows (each will be one batch for one seq-stamp)
    x_list = tf.unstack(x_series, axis=0)
    y_list = tf.unstack(y_series, axis=0)

    # Forward pass
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    states_seq, current_state = tf.contrib.rnn(cell, x_list, init_state)

    # Get the outputs of each state
    logits_seq = [tf.matmul(Why, h) + by for h in states_seq]
    pred_seq = [tf.nn.softmax(logits) for logits in logits_seq]
    losses = [tf.nn.softmax_cross_entropy_with_logits(
        logits=tf.squeeze(logits), labels=y)
        for logits, y in zip(logits_seq, y_list)]
    mean_loss = tf.reduce_mean(losses)
    train_seq = tf.train.AdagradOptimizer(0.5).minimize(mean_loss)

    # Simple single forward pass:
    single_logits = tf.matmul(Why, single_state) + by
    single_predicion = tf.nn.softmax(single_logits, 0)
    single_state_return = tf.tanh(tf.matmul(Whh, single_state)
                                  + tf.matmul(Wxh, single_input)
                                  + bh)
    # Init all variables
    init = tf.global_variables_initializer()


def sample(h, seed_ix, n):
    """
    sample a sequence of n integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    with tf.device('/device:GPU:0'):
        sample = []
        x = np.zeros([vocab_size, 1])
        x[seed_ix] = 1
        for _ in range(n):
            logits, pred, h = sess.run(
                [single_logits, single_predicion, single_state_return],
                feed_dict={
                    single_input: x,
                    single_state: h
                })
            ix = np.random.choice(list(range(vocab_size)), p=np.ravel(pred))
            x = np.zeros([vocab_size, 1])
            x[ix] = 1
            sample.append(ix)
    return sample


config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    sess.run(init)
    loss_list = []

    n, p = 0, 0
    _current_state = np.random.randn(hidden_size, 1)
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= len(data) or n == 0:
            p = 0  # go from start of data
        input_chars_ix = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        target_chars_ix = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
        input_series = np.zeros([seq_length, vocab_size])
        target_series = np.zeros([seq_length, vocab_size])
        for i in range(len(input_chars_ix)):
            input_series[i][input_chars_ix[i]] = 1
            target_series[i][target_chars_ix[i]] = 1

        # sample from the model now and then
        if n % 500 == 0:
            seed_ix = input_chars_ix[0]
            sample_ix = sample(_current_state, seed_ix, 400)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(('----\n %s \n----' % (txt,)))

        # forward seq_length characters through the net and fetch gradient
        _mean_loss, _train_seq, _current_state, _pred_seq = sess.run(
            [mean_loss, train_seq, current_h, pred_seq],
            feed_dict={
                x_series: input_series,
                y_series: target_series,
                init_h: _current_state
            })
        if n % 1000 == 0: print(('iter %d, loss: %f' % (n, _mean_loss)))  # print progress

        p += seq_length  # move data pointer
        n += 1  # iteration counter
