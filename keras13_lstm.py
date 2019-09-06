from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense
#1 데이터
X = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
Y = array([4,5,6,7])


print("X.shape : ",X.shape) # 4,3
print("Y.shape : ",Y.shape) # 4,
# RNN은 입력이 3개다. 몇행 몇열, 몇개씩 작업을 할 것인가. 1도 가능하고 2도 가능하고.
X = X.reshape((X.shape[0],X.shape[1],1)) # 몇행 몇열 + 몇개씩 들어가느냐. 이부분에서는 1개씩 들어간다.
#
print("X.shape : ",X.shape) # 4, 3, 1
print("Y.shape : ",Y.shape) # 4,
min = 10

def Add(LayerCount):
    global min,X,Y
    if LayerCount == 0 :
        for i in range(1,1500):
            for j in range(20):
                model.add(LSTM(3)) # 여기를 input_shape의 열과 맞춰줘야하나봄. 내일 물어보자!
                model.add(Dense(1,activation='relu'))
                model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
                model.fit(X,Y,epochs=i,verbose=2)
                x_input = array([6,7,8])
                x_input = x_input.reshape((1,3,1))
                yhat = model.predict(x_input,verbose=0)
                if(abs(9 - min) > abs(9 - yhat)):
                    min = yhat
                    if(min == 9):
                        break;
            if(min == 9):
                break;
    else :
        for i in range(1,80):
            model.add(LSTM(i,return_sequences=True))
            Add(LayerCount-1)
            if(min == 9):
                break;

if __name__=='__main__':
    #2 모델 구성
    for i in range(1,11):
        model = Sequential()
        model.add(LSTM(20,activation='relu',input_shape=(X.shape[1],X.shape[2]),return_sequences=True)) # input_shape=(3,1) > 3열에 대해 1개씩 한다.
        Add(i)
        model
        if(min == 9):
            break;
    print(min)
    # model.add(LSTM(40,return_sequences=True))
    # model.add(LSTM(30,return_sequences=True))
    # model.add(LSTM(50,return_sequences=True))
    # model.add(LSTM(20,return_sequences=True))
    # model.add(LSTM(3)) # 여기를 input_shape의 열과 맞춰줘야하나봄. 내일 물어보자!
    # model.add(Dense(1,activation='relu'))

    # model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

    # #3 실행
    # model.fit(X,Y,epochs=1400,verbose=2)
    # # demonstrate prediction
    # # x_input = array([6,7,8])
    # x_input = array([70,80,90])
    # x_input = x_input.reshape((1,3,1))
    # yhat = model.predict(x_input,verbose=0)
    # print(yhat)
# while(1):
#     model.fit(X,Y,epochs=1400,verbose=2)
# # demonstrate prediction
#     x_input = array([6,7,8])
#     x_input = x_input.reshape((1,3,1))
#     yhat = model.predict(x_input,verbose=0)
#     if(yhat >= 8.99 and yhat <= 9.01):
#         print(yhat)
#         break;