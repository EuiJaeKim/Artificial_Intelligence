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

 # NaN 값 갯수 확인
# print(train.info())
# print(test.info())
# print(train.isnull().sum())
# print(test.isnull().sum())
# 이걸 보고 NaN값을 볼꺼야. 그래서 뭔가 상관이 있을거 같은 데이터를 처리해.

# bar_chart('Sex') # So, The Chart confirms Women more likely survivied than Men
# bar_chart('Pclass') # The Chart confirms 1st class more likely survivied than other classes
# bar_chart('SibSp') #  The Chart confirms a person aboarded with more than 2 siblings or spouse more likely survived
# bar_chart('Parch') # The Chart confirms a person aboarded with more than 2 parents or children more likely survived
# bar_chart('Embarked')
# The Chart confirms a person aboarded from C slightly more likely survived
# The Chart confirms a person aboarded from Q more likely dead
# The Chart confirms a person aboarded from S more likely dead

train_test_data = [train, test] # combining train and test dataset 실제로 합친건 아니고 각각 연결하는 포인터 개념인듯
# dataset에서 Name에서 mr. mrs. 으로 된거를 가져올꺼야 이름보단 남자인지 여자인지 직위 그런거가 더 연관 있을꺼니까
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# print(train['Title'].value_counts())
# print(test['Title'].value_counts())
# str로 된걸 num으로 맵핑해줄꺼야 그래서 기반을 map 자료구조를 만들고
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
# train_test_data의 Title에 매핑을 때려
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
# bar_chart('Title')
# Name 열 삭제
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# str로 된거 다 num으로 맵핑해줄꺼야~
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
# bar_chart('Sex')
# print(train.head(100))
# print(train.isnull().sum())
# 나이에서 NaN값을 처리해야해 어떻게 할까? 그냥 평균? 아니면 랜덤?(은 하면 이상하게 나오겠지)
# 크게 Title을 기준으로 Title끼리의 Age의 중간값으로 넣어본다.
# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
# 남자는 남자들끼리의 평균 여자는 여자들끼리 평균 잡아보자.
# print('----------------------')
# print(train['Age'])
# print(train.isnull().sum())
# facet = sns.FacetGrid(train, hue="Survived",aspect=4)
# facet.map(sns.kdeplot,'Age',shade= True)
# facet.set(xlim=(0, train['Age'].max()))
# facet.add_legend()
# plt.show()
# 다른 값에 비해서 Age 값의 
# feature vector map:
# child: 0
# young: 1
# adult: 2
# mid-age: 3
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
# print(train.head())
# bar_chart('Age')

# Embarked : 승선지가 어딘지? 승선지에 따라서 1등석, 2등석, 3등석의 분포를 볼 수있을것같음.
# Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
# Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
# Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
# df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
# df.index = ['1st class','2nd class', '3rd class']
# df.plot(kind='bar', figsize=(10,5))
# plt.show()
# 봐보니 어느정도 그러함
# NaN이 2개 있어서 그걸 채워줌.
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# print(train.isnull().sum())
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
# str을 num으로 전환
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

#  요금이 NaN인거가 있어서 채워줌.
# fill missing Fare with median fare for each Pclass
# print(test.isnull().sum())
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
# 티켓 요금에 따라서 비싸게 냈으면 앵간하면 다 살았네?
# facet = sns.FacetGrid(train, hue="Survived",aspect=4)
# facet.map(sns.kdeplot,'Fare',shade= True)
# facet.set(xlim=(0, train['Fare'].max()))
# facet.add_legend()
# plt.show()

# Fare 값이 0~512 여서 치환 해줄꺼임.
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
# facet = sns.FacetGrid(train, hue="Survived",aspect=4)
# facet.map(sns.kdeplot,'Fare',shade= True)
# facet.set(xlim=(0, train['Fare'].max()))
# facet.add_legend()
# plt.show()

# Cabin 선실 번호
# print(train.Cabin.value_counts())
# 맨앞 A,B,C,D 만 뽑아
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
# print(train.Cabin.value_counts())

# 이걸보니 선실의 위치와 좌석과의 연관이 있음.
# Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
# Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
# Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
# df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
# df.index = ['1st class','2nd class', '3rd class']
# df.plot(kind='bar',stacked=True, figsize=(10,5))
# plt.show()
# 맵핑 해주고,
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# NaN 값 채워주고
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

# 가족으로 왔으면 앵간하면 살고 혼자 왔으면 죽었음..
train["FamilySize"] = train["SibSp"] + train["Parch"] +1
test["FamilySize"] = test["SibSp"] + test["Parch"] +1

# facet = sns.FacetGrid(train, hue="Survived",aspect=4)
# facet.map(sns.kdeplot,'FamilySize',shade= True)
# facet.set(xlim=(0, train['FamilySize'].max()))
# facet.add_legend()
# plt.xlim(0)
# plt.show()

# 가족 사이즈 맵핑
family_mapping = {1: 0, 2: 0.5, 3: 1, 4: 1.5, 5: 2, 6: 2.5, 7: 3, 8: 3.5, 9: 4,10: 4.5, 11: 5}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
# 마지막으로 사용 안하는 열 값들 삭제.
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

# print(train_data.shape, target.shape)
# print(train_data.head(10))

# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

# train.info() # Non-NULL

# 교차검증 모든 데이터가 최소 한 번은 테스트셋으로 쓰이도록
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# kNN Score
print(round(np.mean(score)*100, 2))

clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# decision tree Score
print(round(np.mean(score)*100, 2))

clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# Random Forest Score
print(round(np.mean(score)*100, 2))

clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# Naive Bayes Score
print(round(np.mean(score)*100, 2))

clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# SVM Score
print(round(np.mean(score)*100, 2))

# clf = SVC()
# clf.fit(train_data, target)

# test_data = test.drop("PassengerId", axis=1).copy()
# prediction = clf.predict(test_data)

# submission = pd.DataFrame({
#         "PassengerId": test["PassengerId"],
#         "Survived": prediction
#     })

# submission.to_csv('submission.csv', index=False)

# submission = pd.read_csv('submission.csv')
# submission.head()

# -----------------------------------------

# #2. 모델구성
# from keras.models import Sequential
# from keras.layers import Dense
# model = Sequential()

# model.add(Dense(200, input_dim=1, activation='relu'))
# model.add(Dense(3))
# model.add(Dense(100))
# model.add(Dense(4))
# model.add(Dense(1))

# #3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=30, batch_size=3)

# #4. 평가 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=1)
# print("acc : ", acc)

# # y_predict = model.predict(x_test)
# y_predict = model.predict(x4)
# print(y_predict)
