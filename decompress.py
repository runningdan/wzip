import arithmeticcoding
import dictcompress
import sys
import lstm
import numpy as np
import config
import struct
import os

# Command line main application function.
def main(args):

	inputfile, outputfile = args
	total_length = 0

	with open(inputfile, "rb") as inp, open(config.DICT_TEMPFILE, "wb") as out:
		total_length = struct.unpack('>h', inp.read(2))[0]
		bitin = arithmeticcoding.BitInputStream(inp)
		decompress(bitin, out, total_length)

	dict_compress = dictcompress.DictCompress(config.DICT_FILE_PATH)
	dict_compress.decompress(config.DICT_TEMPFILE, outputfile)

def decompress(bitin, out, total_len):

	dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

	decoded_data_series = np.zeros(total_len, dtype = np.uint64)

	# INIT MODEL
	lstm_model = lstm.lstm_model(
		batch_size=config.TIMESTEPS, 
		alphabet_size=config.ALPHABET_SIZE
	)

	# init uniform distrabution
	uniform_prob = np.ones(config.ALPHABET_SIZE)/config.ALPHABET_SIZE
	cummlative_uniform_prob = np.zeros(config.ALPHABET_SIZE+1, dtype = np.uint64)
	cummlative_uniform_prob[1:] = np.cumsum(uniform_prob*10000000+1)       

	for j in range(min(config.TIMESTEPS, total_len)):
		decoded_symbol = dec.read(cummlative_uniform_prob, config.ALPHABET_SIZE)
		print("decoded symbol", decoded_symbol)
		decoded_data_series[j] = decoded_symbol

	if total_len > config.TIMESTEPS:

		for i in range(config.TIMESTEPS, total_len):

			current_steam = decoded_data_series[:i]
			prob_prediction = lstm.predict_next_symbol_from_chunk(current_steam, lstm_model)

			cummulative_probs = np.zeros((config.ALPHABET_SIZE+1), dtype = np.uint64)
			cummulative_probs[1:] = np.cumsum(prob_prediction*10000000+1, axis=1)
					
			decoded_symbol = dec.read(cummulative_probs, config.ALPHABET_SIZE)
			
			print("decoded symbol: ", chr(decoded_symbol))

			decoded_data_series[i] = decoded_symbol

	for decoded_symbol in decoded_data_series:
		out.write(bytes((decoded_symbol,)))


# Main launcher
if __name__ == "__main__":
	main(sys.argv[1 : ])