import os
import numpy as np
import pandas as pd
import keras
from pandas_datareader import data as pddr
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from math import *

class LstmModel():
    def __init__(self,InitSplitSize=5):
        self.MODEL_DIR = './model/'
        if not os.path.exists(self.MODEL_DIR):
            os.mkdir(self.MODEL_DIR)
        self.modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
        self.df = pddr.get_data_yahoo('^GSPC', '1990-01-01') # S&P 500 미국 500대
        # self.LearningRate = 0.005 # 학습률
        self.df = self.df['Close']
        self.df = np.asarray(self.df)
        self.df = self.df.astype(int)
        self.df = np.reshape(self.df,(-1, 1))
        self.DataSplit(self.df,SplitSize=InitSplitSize)
        self.IncDecTrans()
        self.StandardScalerTrans()
        self.TrainTestSplit()

        self.Build_Model()
    def StandardScalerTrans(self):
        scaler = StandardScaler()
        scaler.fit(self.df)
        self._Mean = np.mean(self.df)
        self._Std = np.std(self.df)
        self.X = np.reshape(self.X,(len(self.X),(self.X.shape[1])))
        self.X = scaler.transform(self.X)
        self.X = np.reshape(self.X, (len(self.X),(self.X.shape[1]), 1))
    def IncDecTrans(self):
        for i, val in enumerate(self.X):
            if self.Y[i] > self.X[i][val.size-1]:
                self.Y[i] = 1
            else:
                self.Y[i] = 0
    def RMSE(self,y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
    def DataSplit(self,df,SplitSize=5):
        dataset = []
        for i in range(df.size -(SplitSize+1)):# 열
            subset = df[i:(i+SplitSize)]# 0~SplitSize
            dataset.append([item for item in subset])
        self.X = np.array(dataset)[:,0:SplitSize-1]
        self.Y = np.array(dataset)[:,SplitSize-1]
    def TrainTestSplit(self):
        self.X_test,self.X_train,self.Y_test,self.Y_train = train_test_split(self.X,self.Y,test_size=0.4) # 6 : 4로 나눔
        self.X_test,self.X_validation,self.Y_test,self.Y_validation = train_test_split(self.X_test,self.Y_test,test_size=0.5) # 4를 5:5로 나눔
    def Build_Model(self,SplitSize=5):
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=[self.X_train.shape[1],1], return_sequences=True))
        # self.model.add(LSTM(8, return_sequences=True))
        self.model.add(LSTM(self.X_train.shape[1]))
        # self.model.add(Dense(32,activation='relu'))
        self.model.add(Dense(8,activation='relu'))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.summary()
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    def Train(self,InputEpochs=100 ,InputBatch_size=64):
        self.checkpointer = ModelCheckpoint(filepath=self.modelpath, monitor='val_loss', verbose=1, save_best_only=True)
        self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=InputEpochs/50) # 트레이닝 하다가 10 이상 좋은 값이 안나오면 중단한다.
        self.model_fit = self.model.fit(self.X_train, self.Y_train, epochs=InputEpochs, batch_size=InputBatch_size,
                                        verbose=2, validation_data=(self.X_validation,self.Y_validation),
                                        callbacks=[self.early_stopping_callback,self.checkpointer])
        loss, acc = self.model.evaluate(self.X_test, self.Y_test)
        print('loss : ', loss)
        print('acc : ', acc)
    def Predict(self):
        # 넘파이에서 np.mean 평균값 가져오고 np.std() 표준편차     표준편차를 곱해주고 평균값을 더해준다.
        y_predict = self.model.predict(self.X_test)
        print('y_predict(x_test) : \n', y_predict)
        print("RMSE : ",self.RMSE(self.Y_test, y_predict))
        print("R2 : ", r2_score(self.Y_test, y_predict))
    def Plt_Chart(self):
        self.df.plot()
        # self.model_fit.plot_predict()
        plt.show()
# 'val_loss' 테스트 셋의 오차
# 'loss' 학습셋의 오차
# 'patience' 성능이 증가하지 않는 epoch 을 몇 번이나 허용할 것인가
def AllRemove():
    import os
    file_list = os.listdir('./model/')
    print(file_list)
    for i in file_list:
        if i[0] != '.':
            sstr = './model/' + i
            os.remove(sstr)

if __name__ == "__main__":
    AllRemove()
    lstm = LstmModel(InitSplitSize=22)
    lstm.Train(InputEpochs=1000,InputBatch_size=16)
    lstm.Predict()
    # lstm.Plt_Chart()
