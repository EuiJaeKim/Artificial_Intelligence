import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 현재 실행 하는 위치 확인 할것 PS C:\Users\xcome\Projects>   < 여기서 실행중. os.getcwd()
sns.set()
train = pd.read_csv('./Artificial_Intelligence/Kaggle/Data/Titanic_train.csv',encoding='cp949') # 안읽어지면 인코딩 방식 확인할 것
test = pd.read_csv('./Artificial_Intelligence/Kaggle/Data/Titanic_test.csv',encoding='cp949') 

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show()

train_test_data = [train, test]

# Name에서 특징 뽑아서 Title이란 Column 생성
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# 성을 num화 시킴
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# Age Group화.
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

# Embarked 몇등석? 인지에 대해 num화
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# 요금을 낸것에 대해서 Nan 처리하고 group화
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

# Cabin 좌석의 구역중 크게 A~T까지를 num화 하고 Pclass를 매칭해서 NaN값 채워줌
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

# SibSp이랑 Parch를 더해서 혼자 왔는지 1명이라도 동승자가 있는지 구분하고 num화
train["FamilySize"] = train["SibSp"] + train["Parch"] +1
test["FamilySize"] = test["SibSp"] + test["Parch"] +1
family_mapping = {1: 0, 2: 0.5, 3: 1, 4: 1.5, 5: 2, 6: 2.5, 7: 3, 8: 3.5, 9: 4,10: 4.5, 11: 5}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

# 마지막으로 사용 안하는 열 값들 삭제.
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

# 데이터 전처리 된 원본 = train  테스트 데이터 원본 = test
# train.info()
# test.info()

X = train.drop('Survived', axis=1)
Y = train['Survived']
X_Test = test.drop('PassengerId', axis=1)

# ----------------------------------------

#1. 데이터 
import numpy as np
from sklearn.model_selection import train_test_split

x_test,x_train,y_test,y_train = train_test_split(X,Y,test_size=0.4,random_state=1) # 6 : 4로 나눔
x_test,x_validation,y_test,y_validation = train_test_split(x_test,y_test,test_size=0.5,random_state=1) # 4를 5:5로 나눔

#2. 모델 구성
from keras.layers import Dense, Dropout
from keras.activations import softmax
from keras.models import Sequential
model = Sequential()

# model.add(Dense(20,input_dim = 1, activation = 'relu'))
model.add(Dense(20,input_shape =(X.shape[1],), activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'relu'))

#3. 훈련
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=7,validation_data=(x_validation,y_validation))

#4. 평가 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=1)
print("acc : ",acc)
print("loss : ",loss)

# y_predict = model.predict(x_test)
# print('y_predict : ',y_predict)

# test_data = test.drop("PassengerId", axis=1).copy()
# prediction = clf.predict(test_data)

# submission = pd.DataFrame({
#         "PassengerId": test["PassengerId"],
#         "Survived": prediction
#     })

# submission.to_csv('submission.csv', index=False)

# submission = pd.read_csv('submission.csv')
# submission.head()
