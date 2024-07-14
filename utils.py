import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt

def stride_data(series, timesteps, step_size): 
    nrows = ((series.size - timesteps) // step_size) + 1
    n = series.strides[0]
    return np.lib.stride_tricks.as_strided(
        series, shape=(nrows, timesteps), strides=(step_size * n, n), writeable=False)


def visualize_pred(preds):
    print("DISPLAY PREDICTIONS WITH OVER A 1% PROBIBILITY")
    for chr_num, prob in enumerate(preds[0]):
        pred_val = prob * 100
        if pred_val > 1:
            print(f"CHAR: {str(chr(chr_num))}, PRED: {pred_val}%, CHR NUM: {chr_num}")


def measure_results(input_path: str, output_path: str):
    input_size_bytes = os.path.getsize(input_path)
    output_size_bytes = os.path.getsize(output_path)

    out_str = (
            f"================================================\n"
            f"input size {input_size_bytes}\n"
            f"output output_size_bytes {output_size_bytes}\n"
            f"compression ratio: {(input_size_bytes/output_size_bytes)*100}%\n"
            f"file size reduced bytes: {input_size_bytes-output_size_bytes}\n"
            f"------------------------------------------------\n"
            f"input size bits {input_size_bytes*8}\n"
            f"output output size bits {output_size_bytes*8}\n"
            f"file size reduced bits: {(input_size_bytes*8)-(output_size_bytes*8)}\n"
            f"================================================")

    print(out_str)


class ResultsVisualize:

    def __init__(self):
        self.prediction_results: List[float] = []

    def add_prediction(self, prediction: float) -> None:
        self.prediction_results.append(prediction)

    def show_graph(self) -> None:
        x = []
        for i in range(len(self.prediction_results)):
            x.append(i)

        # define data values
        y_graph = np.array(self.prediction_results) 
        x_graph = np.array(x)

        plt.title('LSTM prediction accuracy')
        plt.xlabel('character position')
        plt.ylabel('prediction accuracy')
        
        # Display grid
        plt.grid(True)
        plt.plot(x_graph, y_graph)  
        plt.show()  

