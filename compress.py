import arithmeticcoding
import contextlib, sys
import lstm
import numpy as np
import config
import struct
import utils
import dictcompress

def main(args):
	inputfile, outputfile = args
	compress_results = CompressResults(encode_length=0)

	dict_compress = dictcompress.DictCompress(config.DICT_FILE_PATH)
	dict_compress.compress(inputfile, config.DICT_TEMPFILE)

	with open(config.DICT_TEMPFILE, "rb") as inp, \
		contextlib.closing(arithmeticcoding.BitOutputStream(open(config.TEMPFILE, "wb"))) as bitout:
		compress_results = compress(inp, bitout)

	with open(outputfile, "wb") as outfile:
		outfile.write(struct.pack('>h', compress_results.encode_length))

		with open(config.TEMPFILE, "rb") as finishedtempfile:
			temp_contents = finishedtempfile.read()
			outfile.write(temp_contents)

	utils.measure_results(inputfile, outputfile)

class CompressResults:
	def __init__(self, encode_length):
		self.encode_length: int = encode_length


def compress(inp, bitout) -> CompressResults:

	# create graph
	visualize_results = utils.ResultsVisualize()

	data_series = np.array([char for char in inp.read()])
	data_series_len = len(data_series)

	print("current len", len(data_series))

	enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

	# INIT MODEL
	lstm_model = lstm.lstm_model(
		batch_size=config.TIMESTEPS, 
		alphabet_size=config.ALPHABET_SIZE
	)
	
	# encode first values with uniform distribution to allow decoding
	# init uniform distrabution
	uniform_prob = np.ones(config.ALPHABET_SIZE)/config.ALPHABET_SIZE
	cummlative_uniform_prob = np.zeros(config.ALPHABET_SIZE+1, dtype = np.uint64)
	cummlative_uniform_prob[1:] = np.cumsum(uniform_prob*10000000+1)       

	# encode first timestep
	for j in range(min(config.TIMESTEPS, data_series_len)):
		enc.write(cummlative_uniform_prob, data_series[j])

	# return early if data is too small to start LSTM prediction process
	if data_series_len <= config.TIMESTEPS:
		enc.finish() 
		return CompressResults(data_series_len)
	
	# encode with adaptive probability distributions
	for i in range(config.TIMESTEPS, len(data_series)):
		# print status
		print((f"encoding character with LSTM {i+config.TIMESTEPS}/{data_series_len}."
		 	  f" {(float((i+config.TIMESTEPS))/float(data_series_len)) * 100 // 1}% complete"))

		current_steam = data_series[:i]
		prob_prediction = lstm.predict_next_symbol_from_chunk(current_steam, lstm_model)
		cummulative_probs = np.zeros((config.ALPHABET_SIZE+1), dtype = np.uint64)
		cummulative_probs[1:] = np.cumsum(prob_prediction*10000000+1, axis=1)

		prediction_accuracy: float = prob_prediction.reshape(-1)[data_series[i]]*100
		visualize_results.add_prediction(prediction_accuracy)
		print(f"real character prediction: {prediction_accuracy}%")

		enc.write(cummulative_probs, data_series[i])
		
	enc.finish()  # Flush remaining code bits

	# show final graph of prediction accuracy
	visualize_results.show_graph()

	return CompressResults(data_series_len)
	
if __name__ == "__main__":
    main(sys.argv[1 : ])