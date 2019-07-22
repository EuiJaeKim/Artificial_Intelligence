#1. 데이터 
import numpy as np
from sklearn.model_selection import train_test_split

XXX=np.transpose(np.array([range(100),range(311,411),range(811,911)]))
XXX2=np.transpose(np.array([range(600,700),range(200,300),range(1000,1100)]))
YYY=np.transpose(np.array([range(100),range(311,411),range(811,911)]))
YYY2=np.transpose(np.array([range(600,700),range(200,300),range(1000,1100)]))

x_test,x_train,y_test,y_train = train_test_split(XXX,YYY,test_size=0.4,random_state=1) # 6 : 4로 나눔
x_test2,x_train2,y_test2,y_train2 = train_test_split(XXX2,YYY2,test_size=0.4,random_state=1) # 6 : 4로 나눔
x_test,x_validation,y_test,y_validation = train_test_split(x_test,y_test,test_size=0.5,random_state=1) # 4를 5:5로 나눔
x_test2,x_validation2,y_test2,y_validation2 = train_test_split(x_test2,y_test2,test_size=0.5,random_state=1) # 4를 5:5로 나눔

# data 정제

#2. 모델 구성
from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.layers.merge import concatenate

first_input = Input(shape=(XXX.shape[1], ))
first_dense = Dense(20,activation = 'relu')(first_input)
first_dense1 = Dense(40)(first_dense)
first_dense2 = Dense(30)(first_dense1)
first_dense3 = Dense(40)(first_dense2)
first_dense4 = Dense(30)(first_dense3)
first_dense5 = Dense(20)(first_dense4)

second_input = Input(shape=(XXX2.shape[1], ))
second_dense = Dense(20,activation = 'relu')(second_input)
second_dense1 = Dense(40)(second_dense)
second_dense2 = Dense(30)(second_dense1)
second_dense3 = Dense(40)(second_dense2)
second_dense4 = Dense(30)(second_dense3)
second_dense5 = Dense(20)(second_dense4)

merge_one = concatenate([first_dense5, second_dense5])

output1 = Dense(10)(merge_one)
output2 = Dense(5)(output1)
merge_two = Dense(5)(output2)

output_1 = Dense(40)(merge_two)
output_12 = Dense(XXX.shape[1])(output_1)
output_2 = Dense(40)(merge_two)
output_22 = Dense(XXX2.shape[1])(output_2)

model = Model(inputs=[first_input, second_input], outputs=[output_12,output_22])

model.summary()

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #기계어로 번역
model.fit([x_train,x_train2],[y_train,y_train2],epochs=500,batch_size=2,validation_data=([x_validation,x_validation2],[y_validation,y_validation2]))

#4. 평가 예측
acc = model.evaluate([x_test,x_test2],[y_test,y_test2],batch_size=1) # loss 함수가 몇인지, 정확도가 몇인지.
print("acc : ",acc)

y_predict = model.predict([x_test,x_test2])
print(y_predict)

# ↓ 숙제

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
RMSE1 = RMSE(y_test,y_predict[0])
RMSE2 = RMSE(y_test2,y_predict[1])
print("RMSE : ",RMSE1) # y_predict는 x_test로 만들어짐. 그래서 y_test를 해줘야 비교를 함.
print("RMSE : ",RMSE2) # y_predict는 x_test로 만들어짐. 그래서 y_test를 해줘야 비교를 함.
print("RMSE AVG : ",(RMSE1+RMSE2)/2)
# R2 (알스퀘어) 결정계수
# 통계학에서, 결정계수는 추정한 선형 모형이 주어진 자료에 적합한 정도를 재는 척도이다. 반응 변수의 변동량 중에서 적용한 모형으로 설명가능한 부분의 비율을 가리킨다. 결정계수의 통상적인 기호는 R2이다.
from sklearn.metrics import r2_score
R21 = r2_score(y_test,y_predict[0])
R22 = r2_score(y_test2,y_predict[1])
print("R2 : ", R21)
print("R2 : ", R22)
print("R2 AVG : ",(R21+R22)/2)