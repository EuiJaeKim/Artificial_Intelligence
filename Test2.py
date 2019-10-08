import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pandas_datareader import data as pddr
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


class LstmModel():
    def __init__(self):
        data = pddr.get_data_yahoo('^GSPC', '2004-01-01') # S&P 500 미국 500대
        self.X = data['close']
        print(self.X)
        pass

if 