#1. 데이터 
import numpy as np

x_train = np.arange(1,11,1)
y_train = np.arange(1,11,1)
x_test = np.arange(11,21,1)
y_test = np.arange(11,21,1)
x_test2 = np.arange(101,111,1)
x_test3 = np.arange(101,121,1)
#2. 모델 구성
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()

model.add(Dense(2000000,input_dim = 1, activation = 'relu'))
model.add(Dense(8))
model.add(Dense(100))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy']) #기계어로 번역
model.fit(x_train,y_train,epochs=30,batch_size=2) #epochs 훈련 횟수 batch_size 입력값을 몇개로 잘라서 넣을꺼니? 1<= N <= x.size() -> default = 32 

# 입력 데이터 갯수 / batch_size * epochs = 총 작업 횟수
#4. 평가 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=1) # loss 함수가 몇인지, 정확도가 몇인지.
print("acc : ",acc)

y_predict = model.predict(x_test3)
print(y_predict)