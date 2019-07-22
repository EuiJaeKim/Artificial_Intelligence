#1. 데이터 
import numpy as np

x_train = np.arange(1,41,1)
y_train = np.arange(1,41,1)
x_test = np.arange(11,51,1)
y_test = np.arange(11,51,1)
#2. 모델 구성
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()

model.add(Dense(2000000,input_dim = 1, activation = 'relu'))
model.add(Dense(17))
model.add(Dense(19))
model.add(Dense(17))
model.add(Dense(13))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #기계어로 번역
# model.fit(x_train,y_train,epochs=500,batch_size=2) #epochs 훈련 횟수 batch_size 입력값을 몇개로 잘라서 넣을꺼니? 1<= N <= x.size() -> default = 32 
model.fit(x_train,y_train,epochs=500) #epochs 훈련 횟수 batch_size 입력값을 몇개로 잘라서 넣을꺼니? 1<= N <= x.size() -> default = 32 

# 입력 데이터 갯수 / batch_size * epochs = 총 작업 횟수
#4. 평가 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=1) # loss 함수가 몇인지, 정확도가 몇인지.
print("acc : ",acc)