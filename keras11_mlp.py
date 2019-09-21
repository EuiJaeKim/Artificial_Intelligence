# Multi Layer Perceptron

#1. 데이터 
import numpy as np
from sklearn.model_selection import train_test_split

XXX=np.transpose(np.array([range(100),range(311,411),range(811,911)]))
YYY=np.transpose(np.array([range(100),range(311,411),range(811,911)]))
print(XXX.shape)
print(XXX.shape[0])
print(XXX.shape[1])
# print(YYY.shape)

x_test,x_train,y_test,y_train = train_test_split(XXX,YYY,test_size=0.4,random_state=1) # 6 : 4로 나눔
x_test,x_validation,y_test,y_validation = train_test_split(x_test,y_test,test_size=0.5,random_state=1) # 4를 5:5로 나눔

# print(XXX.shape) -> (100,) 1차원 100행 1열.
# print(XXX.ndim) -> XXX의 차원

#2. 모델 구성
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()

# model.add(Dense(20,input_dim = 1, activation = 'relu'))
model.add(Dense(20,input_shape =(XXX.shape[1],), activation = 'relu'))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(XXX.shape[1])) # 출력도 2로 변경.  ndim -> XXX의 차원

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #기계어로 번역
# model.compile(loss='mae',optimizer='adam',metrics=['mae']) #기계어로 번역

# Validation 1,2,3,4,5,6,7,8,9,10 의 데이터가 있을 때, 1~5까지는 training 데이터로 주고 6,7은 머신이 test 할 수 있는 데이터로 주고, 8,9,10은 사람이 확인하기 위해 test 값으로 설정 하는것.
# 정확도를 더 높이기 위한..!
# 아래는 머신이 검사하는 데이터랑 사람이 검사하는 데이터랑 같기 때문에 값은 잘나오지만 다른 값을 넣었을 때 어떻게 될지 모름! 그렇기때문에 바꿔줘야해!
# model.fit(x_train,y_train,epochs=2000,batch_size=2,validation_data=(x_test,y_test)) #epochs 훈련 횟수 batch_size 입력값을 몇개로 잘라서 넣을꺼니? 1<= N <= x.size() -> default = 32 
model.fit(x_train,y_train,epochs=10,batch_size=2,validation_data=(x_validation,y_validation))
# 입력 데이터 갯수 / batch_size * epochs = 총 작업 횟수
#4. 평가 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=1) # loss 함수가 몇인지, 정확도가 몇인지.
print("acc : ",acc)
print("loss : ",loss)

y_predict = model.predict(x_test)
print('y_predict : ',y_predict)
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