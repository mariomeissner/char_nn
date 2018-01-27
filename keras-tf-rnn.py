import numpy as np
import re

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.models import load_model


# method for generating text
def generate_text(model, length, VOCAB_SIZE, ix_to_char):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ''.join(y_char)


# method for preparing the training data


# hyperparameters
DATA_DIR = './samples/star_wars.txt'
BATCH_SIZE = 50
HIDDEN_DIM = 100
SEQ_LENGTH = 50
WEIGHTS = 'keras-model.h5'
MODE = 'training'
GENERATE_LENGTH = 150
LAYER_NUM = 2
NUM_EPOCH = 30

# Creating training data
# simplify text data for faster processing
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
model = load_model('weights.best.h5')

# Compile neural net
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
filepath = "weights.best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, verbose=1, mode='max')
# model.save()
callback = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
i = 0
while i < NUM_EPOCH:
    print("\n\nEpoch:", i + 1)
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=1, callbacks=[callback], validation_split=0.2)
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    i = i + 1
