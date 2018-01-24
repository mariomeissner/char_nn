"""
Minimal character-level Vanilla RNN model.
Author: Mario Meissner
"""

import numpy as np
import tensorflow as tf

# data I/O
data = open('samples/arvix_abstracts.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(('data has %d characters, %d unique.' % (data_size, vocab_size)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 500  # size of hidden layer of neurons
seq_length = 50  # number of steps to unroll the RNN for
batch_size = 100  # Number of inputs to evaluate simultaniously
learning_rate = 2e-1

# create the TF graph
with tf.device('/device:GPU:0'):
    Wxh = tf.Variable(np.random.randn(hidden_size, 1) * 0.01, dtype=tf.float32)  # input to hidden
    Whh = tf.Variable(np.random.randn(hidden_size, hidden_size) * 0.01, dtype=tf.float32)  # hidden to hidden
    Why = tf.Variable(np.random.randn(vocab_size, hidden_size) * 0.01, dtype=tf.float32)  # hidden to output
    bh = tf.Variable(tf.zeros((hidden_size, 1)), dtype=tf.float32)  # hidden bias
    by = tf.Variable(tf.zeros((vocab_size, 1)), dtype=tf.float32)  # output bias
    # Each row will be a batch
    x_batches = tf.placeholder(tf.float32, (seq_length, batch_size))  # our input
    y_batches = tf.placeholder(tf.int32, (seq_length, batch_size))  # our expected output label
    single_state = tf.placeholder(tf.float32, (hidden_size, 1))
    init_h = tf.placeholder(tf.float32, (hidden_size, batch_size))
    single_input = tf.placeholder(tf.float32, 1)

    # Unpack the rows (each will be one batch for one seq-stamp)
    x_batches_seq = tf.unstack(x_batches, axis=0)
    y_batches_seq = tf.unstack(y_batches, axis=0)

    # Unroll graph for seq_length timesstamps
    h_seq = list()
    current_h = init_h
    for x_batch in x_batches_seq:
        # Add the second dimension back (x_batch will be a column)
        x_batch = tf.expand_dims(x_batch, 0)
        next_h = tf.tanh(tf.matmul(Whh, current_h) +
                         tf.matmul(Wxh, x_batch) + bh)  # bh is a broadcasted summation
        h_seq.append(next_h)
        current_h = next_h

    # Get the outputs of each state
    logits_seq = [tf.matmul(Why, h) + by for h in h_seq]
    pred_seq = [tf.nn.softmax(logits) for logits in logits_seq]
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.transpose(logits), labels=y_batches)
        for logits, y_batches in zip(logits_seq, y_batches_seq)]
    mean_loss = tf.reduce_mean(losses)
    train_seq = tf.train.AdagradOptimizer(0.3).minimize(mean_loss)

    # Simple single forward pass:
    single_logits = tf.matmul(Why, single_state) + by
    single_predicion = tf.nn.softmax(single_logits, 0)
    single_state_return = tf.tanh(tf.matmul(Whh, single_state)
                                  + tf.matmul(Wxh, tf.expand_dims(single_input, 0))
                                  + bh)
    # Init all variables
    init = tf.global_variables_initializer()


def sample(h, seed_ix, n):
    """
    sample a sequence of n integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    sample = []

    h = np.expand_dims(h, 1)
    ix = seed_ix
    for _ in range(n):
        logits, pred, h = sess.run(
            [single_logits, single_predicion, single_state_return],
            feed_dict={
                single_input: [ix],
                single_state: h
            })
        ix = np.random.choice(list(range(vocab_size)), p=np.ravel(pred))
        sample.append(ix)
    return sample


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.Session(config=config) as sess:
    sess.run(init)
    loss_list = []

    n, p = 0, 0
    _current_state = np.random.randn(hidden_size, batch_size)
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length * batch_size + 1 >= len(data) or n == 0:
            p = 0  # go from start of data
        input_chars_ix = [char_to_ix[ch] for ch in data[p:p + seq_length * batch_size]]
        target_chars_ix = [char_to_ix[ch] for ch in data[p + 1:p + seq_length * batch_size + 1]]

        input_batches = np.reshape(input_chars_ix, [seq_length, batch_size])
        target_batches = np.reshape(target_chars_ix, [seq_length, batch_size])
        # we need to create the input and target arrays of shape vocabsize x seq_length
        # input_oneshots = np.zeros((vocab_size, seq_length))
        # target_oneshots = np.zeros((vocab_size, seq_length))
        # for t in range(seq_length):
        #    input_oneshots[input_chars_ix[t]][t] = 1  # set the char position to 1
        #    target_oneshots[target_chars_ix[t]][t] = 1  # set the char position to 1

        # sample from the model now and then
        if n % 500 == 0:
            seed_ix = input_batches[0, 0]
            sample_ix = sample(_current_state[:, 0], seed_ix, 400)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(('----\n %s \n----' % (txt,)))

        # forward seq_length characters through the net and fetch gradient
        _mean_loss, _train_seq, _current_state, _pred_seq = sess.run(
            [mean_loss, train_seq, current_h, pred_seq],
            feed_dict={
                x_batches: input_batches,
                y_batches: target_batches,
                init_h: _current_state
            })
        if n % 1000 == 0: print(('iter %d, loss: %f' % (n, _mean_loss)))  # print progress

        p += seq_length  # move data pointer
        n += 1  # iteration counter
