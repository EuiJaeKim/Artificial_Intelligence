#1. 데이터 
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델 구성
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()

model.add(Dense(1,input_dim = 1, activation = 'relu'))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(1000))
model.add(Dense(2000))
model.add(Dense(3000))
model.add(Dense(2000))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #기계어로 번역
model.fit(x,y,epochs=150,batch_size=1) #300번 훈련시켜라 총 900번이래

#4. 평가 예측
loss, acc = model.evaluate(x,y,batch_size=1) # loss 함수가 몇인지, 정확도가 몇인지.
print("acc : ",acc)