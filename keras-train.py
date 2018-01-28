import numpy as np
import re

import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential


# method for generating text
def generate_text(model, length, VOCAB_SIZE, index_to_char):
    index = [np.random.randint(VOCAB_SIZE)]
    y_char = [index_to_char[index[-1]]]
    print("\ny_char:", y_char)
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][index[-1]] = 1
        print(index_to_char[index[-1]], end="")
        index = np.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_char.append(index_to_char[index[-1]])
    return ''.join(y_char)


# method for preparing the training data


# hyperparameters
DATA_DIR = './samples/star_wars.txt'
BATCH_SIZE = 50
HIDDEN_DIM = 100
SEQ_LENGTH = 50
GENERATE_LENGTH = 50
LAYER_NUM = 2
NUM_EPOCH = 1

# Parsing training data
# Simplify text for faster processing
data = re.sub('[^a-z\s]', '', re.sub('\s+', ' ', open(DATA_DIR, 'r').read()).lower())

chars = list(set(data))
DATA_SIZE, VOCAB_SIZE = len(data), len(chars)

print('data has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE))

ix_to_char = {ix: char for ix, char in enumerate(chars)}
char_to_ix = {char: ix for ix, char in enumerate(chars)}

NUM_OF_SEQUENCES = int(len(data) / SEQ_LENGTH)

X = np.zeros((NUM_OF_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((NUM_OF_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))

for i in range(0, NUM_OF_SEQUENCES):
    X_sequence = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence

    y_sequence = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
        y[i] = target_sequence

# Creating neural net
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))

# Compile neural net
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='./weights.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                            write_graph=True, write_images=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
# model.save()

i = 0
while i < NUM_EPOCH:
    print("\n\nEpoch:", i + 1)
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=50, callbacks=[checkpointer, tbCallBack], validation_split=0.2)
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    model.save('starwars-model.h5')
    i = i + 1
