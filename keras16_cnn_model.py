from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,Flatten,MaxPooling2D

filter_size = 16
kernel_size = (3,3)
model = Sequential() 

# filter_size > 출력
# kernel_size > 잘라서 볼 크기
model.add(Conv2D(filter_size,kernel_size, padding = 'same', # padding ='valid' 기본값인데 output의 크기를 건드지리 않음. 'same'을 주면 테두리에 0을 채워서 크기를 유지 시킴.
                 input_shape = (7,7,1))) # (None, 6, 6, 32) 7x7 사진을 2x2로 나누면 6x6으로 나오고 출력은 32개
model.add(Conv2D(16,(2,2))) # (None, 4, 4, 16) 6x6 사진을 3x3로 나누면 4x4으로 나오고 출력은 16개
model.add(Conv2D(100,(3,3))) # (None, 2, 2, 100)  4x4 사진을 3x3로 나누면 2x2으로 나오고 출력은 100개
model.add(MaxPooling2D(pool_size=2)) # 2x2씩 짤라보면서 총4개의 값중 Max 값만 남기고 나머지는 버린다. (특이점만 남김. 각져지거나 진해지거나. 그런 특성이 나옴. 그럼 특징을 잡는데 더 쉬워짐)

model.add(Flatten()) # (None, 400) > 일렬로 데이터를 만듬. 이후부터는 그냥 dense로 처리.

# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 7, 7, 16)          160    > 16,(3,3),(7,7,1)  16 * ((3*3)+1)  = 160
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 6, 6, 16)          1040   > 16&1,(2,2),(7,7,16&2) ((16&2 *(2*2))+1) * 16&1 = 1040
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 4, 4, 100)         14500 > 100,(3,3),(6,6,16)      ((16*(3*3))+1) * 100 =  14500

# Param 계산법 : ((이전층 출력갯수 * 지금필터갯수) + bias) * 지금층의 출력갯수

# model.add(Dense(128,  activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# from keras.layers import MaxPool2D
# pool_size = (2,2)
# model.add(MaxPool2D(pool_size))

model.summary()