#Currently does not work. The model and weights are not able to loaded in a seperate python instance.


# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import csv
# import re
# import os
# import keras
# from pathlib import Path
# from keras import optimizers
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM, SimpleRNN
# from keras.layers.wrappers import TimeDistributed
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import load_model
# from keras.callbacks import ModelCheckpoint
# from keras.models import model_from_json
#
# def generate_text(model, length, VOCAB_SIZE, index_to_char):
#
#     index = [np.random.randint(VOCAB_SIZE)][-1]
#     print(index)
#     y_char = [index_to_char[index]]
#     print("\ny_char:",y_char)
#     X = np.zeros((1, length, VOCAB_SIZE))
#     for i in range(length):
#         X[0, i, :][index] = 1
#         print(index_to_char[index], end="")
#         index = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
#         y_char.append(index_to_char[index])
#     return ('').join(y_char)
#
# DATA_DIR = './samples/star_wars.txt'
# BATCH_SIZE = 50
# HIDDEN_DIM = 200
# SEQ_LENGTH = 50
# WEIGHTS = 'keras-model.h5'
# MODE = 'training'
# GENERATE_LENGTH = 100
# LAYER_NUM = 2
# NUM_EPOCH = 1
#
# #Parsing training data
# #Simplify text for faster processing
# data = re.sub('[^a-z\s]','',re.sub('\s+',' ',open(DATA_DIR, 'r').read()).lower())
# chars = list(set(data))
# DATA_SIZE, VOCAB_SIZE = len(data), len(chars)
#
# print('data has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE))
#
# ix_to_char = {ix:char for ix, char in enumerate(chars)}
# char_to_ix = {char:ix for ix, char in enumerate(chars)}
#
# NUM_OF_SEQUENCES = int(len(data)/SEQ_LENGTH)
#
# X = np.zeros((NUM_OF_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))
# y = np.zeros((NUM_OF_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))
#
# for i in range(0, NUM_OF_SEQUENCES):
# 	X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
# 	X_sequence_ix = [char_to_ix[value] for value in X_sequence]
# 	input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
# 	for j in range(SEQ_LENGTH):
# 		input_sequence[j][X_sequence_ix[j]] = 1.
# 		X[i] = input_sequence
#
# 	y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
# 	y_sequence_ix = [char_to_ix[value] for value in y_sequence]
# 	target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
# 	for j in range(SEQ_LENGTH):
# 		target_sequence[j][y_sequence_ix[j]] = 1.
# 		y[i] = target_sequence
