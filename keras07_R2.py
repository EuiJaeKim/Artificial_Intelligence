#1. 데이터 
import numpy as np

x_train = np.arange(1,11,1)
y_train = np.arange(1,11,1)
x_test = np.arange(1001,1011,1)
y_test = np.arange(1001,1011,1)
# x_test2 = np.arange(101,111,1)
#2. 모델 구성
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()

model.add(Dense(20,input_dim = 1, activation = 'relu'))
model.add(Dense(40))
model.add(Dense(5))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #기계어로 번역
# model.compile(loss='mae',optimizer='adam',metrics=['mae']) #기계어로 번역
model.fit(x_train,y_train,epochs=2000,batch_size=2) #epochs 훈련 횟수 batch_size 입력값을 몇개로 잘라서 넣을꺼니? 1<= N <= x.size() -> default = 32 

# 입력 데이터 갯수 / batch_size * epochs = 총 작업 횟수
#4. 평가 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=1) # loss 함수가 몇인지, 정확도가 몇인지.
print("acc : ",acc)
print("loss : ",loss)
 
y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict)) # y_predict는 x_test로 만들어짐. 그래서 y_test를 해줘야 비교를 함.

# R2 (알스퀘어) 결정계수
# 통계학에서, 결정계수는 추정한 선형 모형이 주어진 자료에 적합한 정도를 재는 척도이다. 반응 변수의 변동량 중에서 적용한 모형으로 설명가능한 부분의 비율을 가리킨다. 결정계수의 통상적인 기호는 R2이다.
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("R2 : ", r2_y_predict)