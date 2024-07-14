python generatelstmweights.py trainingdata/data.txt trainingdata/temp-dict-preprocess.txt dictionary/tokens.dic
python compress.py testinputs/rawin.txt testinputs/compressed.wzip
python decompress.py testinputs/compressed.wzip testinputs/out.txt