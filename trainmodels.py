import utils
import sys
import numpy as np
import config
import keras
import lstm
import dictcompress
import nltk


nltk.download('punkt')

def generate_tokens(input_file: str, dict_file_path: str) -> None:
    with open(input_file, "r") as inp:
        tokens = nltk.word_tokenize(inp.read())
        grams_distrabution = nltk.FreqDist(nltk.ngrams(tokens, 1)).most_common(1000)
    
    with open(dict_file_path, "w") as out_dict:
        for gram in grams_distrabution:
            detokenized = gram[0][0]
            if len(detokenized) > 3:
                out_dict.write(detokenized + "\n")
        
def main(args):
    
    training_data_file, training_temp_file, dict_file_path = args

    print("generating grams")
    generate_tokens(training_data_file, dict_file_path)

    print("applying dictionary compression on training data")
    dict_compress = dictcompress.DictCompress(config.DICT_FILE_PATH)
    dict_compress.compress(training_data_file, training_temp_file)

    print("starting training LSTM")
    with open(training_temp_file, "rb") as inp:
        data_series = np.array([char for char in inp.read()])

    strided_data_series = utils.stride_data(data_series, config.TIMESTEPS, 1)

    X = strided_data_series[:, :-1]
    Y = strided_data_series[:, -1:]

    # one hot encode data to feed lstm
    x_train = keras.utils.to_categorical(X, config.ALPHABET_SIZE)
    y_train = keras.utils.to_categorical(Y, config.ALPHABET_SIZE)

    model = lstm.lstm_model(
        batch_size=config.TIMESTEPS, 
		alphabet_size=config.ALPHABET_SIZE
    )
    model.fit(x_train, y_train,
        batch_size=128,
        epochs=config.LSTM_EPOCHS,
        verbose=1
    ) 
    model.save(config.WEIGHTS_OUTPUT, overwrite=True)

if __name__ == "__main__":
    main(sys.argv[1 : ])