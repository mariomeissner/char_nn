"""
Minimal character-level Vanilla RNN model.
Heavily modified version of the min-char-nn gist of Andrew Carpathy.
Mario Meissner
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
learning_rate = 2e-1

# model parameters
Wxh = tf.Variable(np.random.randn(hidden_size, vocab_size) * 0.01)  # input to hidden
Whh = tf.Variable(np.random.randn(hidden_size, hidden_size) * 0.01)  # hidden to hidden
Why = tf.Variable(np.random.randn(vocab_size, hidden_size) * 0.01)  # hidden to output
bh = tf.Variable(tf.zeros((hidden_size, 1)))  # hidden bias
by = tf.Variable(tf.zeros((vocab_size, 1)))  # output bias
x = tf.placeholder(tf.float32, vocab_size)  # our input
y = tf.placeholder(tf.float32, vocab_size)  # our expected output label
hprev = tf.placeholder(tf.floar32, hidden_size, hidden_size)  # TODO: Check this?
h = tf.tanh(tf.Variable(tf.matmul(Whh, hprev) +
                        tf.matmul(Wxh, x, b_is_sparse=True) + bh))  # TODO: Check this?
init = tf.initialize_all_variables()
logits = tf.matmul(Why, h) + by


def sample(h, seed_ix, n):
    """
    sample a sequence of n integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    #  TODO: Rewrite this using tf
    return


n, p = 0, 0
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # sample from the model now and then
    if n % 500 == 0:
        seed_ix = inputs[0]
        sample_ix = sample(hprev, seed_ix, 400)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(('----\n %s \n----' % (txt,)))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_fun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 1000 == 0: print(('iter %d, loss: %f' % (n, smooth_loss)))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter



