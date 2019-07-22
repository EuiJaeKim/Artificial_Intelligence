#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data() # mnist라는 예제가 있고, 이 예제를 써서 공부해라!
# print(X_train)
# print(X_train.shape) # (60000, 28, 28)
# print(X_test.shape) # (10000, 28, 28)
# print(Y_train.shape) # (60000,)
# print(Y_test.shape) # (10000,)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(X_train)
print(X_train.shape) # (60000, 28, 28, 1)
print(X_test.shape) # (10000, 28, 28, 1)
print(Y_train.shape) # (60000, 10)  # 회기모델이 아니라 분류 모델이라 숫자가 1~10까지라 10인거
print(Y_test.shape) # (10000, 10)

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu')) # (None, 26, 26, 32)        320
model.add(Conv2D(64, (3, 3), activation='relu')) # (None, 24, 24, 64)        18496
model.add(MaxPooling2D(pool_size=2)) # (None, 12, 12, 64)        0
model.add(Dropout(0.25)) # 노드 수의 경량화를 노리면서 과적합을 살펴보고, dropout을 작성한 이전 층에만 적용된다. 노드의 수 %만큼 black.
model.add(Flatten()) # (None, 9216)
model.add(Dense(128,  activation='relu')) # (None, 128)               1179776    # relu 는 회기
model.add(Dropout(0.5)) 
model.add(Dense(10, activation='softmax')) # (None, 10)                1290      # softmax 는 분류 0부터 9까지!
model.summary()

model.compile(loss='categorical_crossentropy', #  activation='softmax' 이면 loss='categorical_crossentropy' 얘가 잘나옴(항상 그런건 아님)
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10) # 트레이닝 하다가 10 이상 좋은 값이 안나오면 중단한다.

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])
# Scikit-learn 에서도 fit을 씀

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()