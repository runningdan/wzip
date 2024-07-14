import keras
from typing import Any
import config 
import numpy as np

import os
import random
import tensorflow as tf
import numpy as np

# reduce randomness to allow for correct decoding
np.random.seed(1000)
os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
tf.random.set_seed(1234)

def predict_next_symbol_from_chunk(char_stream, model) -> np.array:
	final_chunk = char_stream[-config.TIMESTEPS+1:].reshape(1, -1)
	prob_prediction = model.predict(keras.utils.to_categorical(final_chunk, config.ALPHABET_SIZE))
	return prob_prediction

def lstm_model(batch_size, alphabet_size) -> keras.Sequential:
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(batch_size, alphabet_size)))
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dense(alphabet_size, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=keras.optimizers.RMSprop(learning_rate=0.01)
    )
    model.load_weights(config.WEIGHTS_OUTPUT)
    model.summary()
    return model
