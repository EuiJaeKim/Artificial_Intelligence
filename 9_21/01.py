import numpy as np
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

# x = np.array(range(1,51))
# y = np.array(range(1,51))

x = np.array([range(100),range(311,411),range(100)])
y = np.array([range(100),range(311,411),range(100)])

x_test,x_train,y_test,y_train = train_test_split(x,y,test_size=0.4,random_state=1) # 6 : 4로 나눔
x_test,x_validation,y_test,y_validation = train_test_split(x_test,y_test,test_size=0.5,random_state=1) # 4를 5:5로 나눔

# print(x.shape)
# print(y.shape)
# print(x,y)

model = Sequential()
model.add(Dense(32, input_shape = (x.shape[1],), activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(y.shape[1]))

# 함수형 Start
# from keras.models import Model
# from keras.layers import Input

# input1 = Input(shape=(x.shape[1],))
# dense1 = Dense(32,activation = 'relu')(input1)
# dense2 = Dense(8,activation = 'relu')(dense1)
# output1 = Dense(y.shape[1])(dense2)

# model = Model(inputs = input1, outputs = output1)
# 함수형 End

model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=1100,batch_size=2)
# model.fit(x_train,y_train,epochs=1100,batch_size=2,validation_data=(x_validation,y_validation))

loss, acc = model.evaluate(x_test,y_test,batch_size=1)
print("acc : ",acc)
print("loss : ",loss)

y_predict = model.predict(x_test)
print(y_predict)
print("RMSE : ",RMSE(y_test,y_predict))

r2_y_predict = r2_score(y_test,y_predict)
print("R2 : ", r2_y_predict)