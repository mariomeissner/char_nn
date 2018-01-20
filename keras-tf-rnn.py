import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import re
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model

# method for generating text
def generate_text(model, length, VOCAB_SIZE, ix_to_char):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

# method for preparing the training data


#hyperparameters
DATA_DIR = './samples/star_wars.txt'
BATCH_SIZE = 50
HIDDEN_DIM = 100
SEQ_LENGTH = 50
WEIGHTS = ''
MODE = 'training'
GENERATE_LENGTH = 100
LAYER_NUM = 2
NUM_EPOCH = 10

# Creating training data

#simplify text data for faster processing
data = re.sub('[^a-z\s]','',re.sub('\s+',' ',open(DATA_DIR, 'r').read()).lower())

chars = list(set(data))
DATA_SIZE, VOCAB_SIZE = len(data), len(chars)

print('data has %d characters, %d unique.' % (DATA_SIZE, VOCAB_SIZE))

ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

NUM_OF_SEQUENCES = int(len(data)/SEQ_LENGTH)

X = np.zeros((NUM_OF_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((NUM_OF_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))

for i in range(0, NUM_OF_SEQUENCES):
	X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
	X_sequence_ix = [char_to_ix[value] for value in X_sequence]
	input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
	for j in range(SEQ_LENGTH):
		input_sequence[j][X_sequence_ix[j]] = 1.
		X[i] = input_sequence

	y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
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

#Compile neural net
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])

#TODO
#check to see if previous weights should be loaded
if not WEIGHTS == '':
  # print('Loading weights')
  # # load json and create model
  # del model
  # model = load_model('keras-model.h5')
  # print("Loaded model from disk")
  # generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
  # print('\n\n')
else:
  NUM_EPOCH = 1

# Training if there is no trained weights specified
if MODE == 'training' or WEIGHTS == '':
	while True:
		print('\n\nEpoch: {}\n'.format(NUM_EPOCH))
		model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
		NUM_EPOCH += 1
		generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
		if NUM_EPOCH % 10 == 0:
			# serialize model to JSON
			model_json = model.to_json()
			with open("keras-model.json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model.save("keras-model.h5")
			print("Saved model and weights to disk")

#load the weights and get text
elif WEIGHTS != '':
	# Loading the trained weights
	print('Loading weights')
	# load json and create model
	json_file = open('keras-model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("keras-model.h5")
	print("Loaded model from disk")
	generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
	print('\n\n')
else:
	print('\n\nNothing to do!')
