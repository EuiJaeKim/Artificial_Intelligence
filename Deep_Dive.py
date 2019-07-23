## 비정상 거래 검출 모델(Fraud Detection System)

#kaggle 제공

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#데이터 로드
data = pd.read_csv('/content/drive/My Drive/data/creditcard.csv')
print(data.head())
print(data.columns)

#데이터 빈도수 학인
print(pd.value_counts(data['Class']))

pd.value_counts(data['Class']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

### 비정상 거래 검출 모델(Fraud Detection System (2)

sdscaler = StandardScaler()
data['normAmount'] = sdscaler.fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis = 1)
print(data.head())

### 비정상 거래 검출 모델(Fraud Detection System (3)

X = np.array(data.ix[:, data.columns != 'Class'])#독립변수
y = np.array(data.ix[:, data.columns == 'Class'])#종속변수
print('Shape of X : {}'.format(X.shape))
print('Shape of y : {}'.format(y.shape))

# 데이터 train, test 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("Number transactions X train dataset: ", X_train.shape)
print("Number transactions y train dataset: ", y_train.shape) 
print("Number transactions X_test dataset:", X_test.shape) 
print("Number transactions y test dataset: ", y_test.shape)

### 비정상 거래 검출 모델(Fraud Detection System (4)

# 데이터 불균형 맞추기 
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} Wn".format(sum(y_train==0))) 
print("y_train",y_train) 
print("y_train.ravel",y_train.ravel())

### 비정상 거래 검출 모델(Fraud Detection System (5)

sm = SMOTE (random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} Wn'.format(y_train_res.shape)) 
print("After OverSampling, counts of y_ train_res '1': {}".format(sum(y_train_res==1))) 
print("After OverSampling, counts of y_train_res 'O': {}Wn".format(sum(y_train_res==0)))

print('After OverSampling, the shape of train_X: {}'.format(X_test.shape))
print('After OverSampling, the shape of train_y: {}Wn'.format(y_test.shape)) 
# 실제 정확도를 알아보기위한 새로운 데이터 갯수 
print("before OverSampling, counts of label '1': {}".format(sum(y_test ==1)))
print("before OverSampling, counts of label '0': {}".format(sum(y_test== 0)))


### 비정상 거래 검출 모델(Fraud Detection System (6)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score 
from sklearn.svm import SVC

#X_train, X_test, y train, y_test X_train_res,y_train_res 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score 
import seaborn as sns; sns.set 
import warnings 
warnings.filterwarnings("ignore")

### 비정상 거래 검출 모델(Fraud Detection System (7)

def logisticR(X_train, y_train, X_test, y_test, disc):
  lr = LogisticRegression()
  lr.fit(X_train , y_train.ravel()) #resample 전 모델 학습
  y_test_pre = lr.predict(X_test)
  
  print(disc + 'accuracy_score : {:.2f}%'.format(accuracy_score(y_test, y_test_pre) * 100))
  print(disc + 'recall_score : {:.2f}%'.format(recall_score(y_test, y_test_pre)*100))
  print(disc + 'precision_score : {:.2f}%'.format(precision_score(y_test, y_test_pre)*100))
  print(disc + 'roc_auc_score : {:.2f}%'.format(roc_auc_score(y_test, y_test_pre)*100))
 
 
 ### 비정상 거래 검출 모델(Fraud Detection System (8)

def logisticR2(X_train ,y_train , X_test, y_test, disc):
  lr = LogisticRegression()
  lr.fit(X_train, y_train.ravel()) #resample 전 모델 학습
  y_test_pre = lr.predict(X_test)
  
  cnf_matrix = confusion_matrix(y_test, y_test_pre)
  print(disc + '===>\n', cnf_matrix)  #matrix 갯수
  print('cnf_matrix_test[0,0]>=',cnf_matrix[0,0])
  print('cnf_matrix_test[0,1]>=',cnf_matrix[0,1])
  print('cnf_matrix_test[1,0]>=',cnf_matrix[1,0])
  print('cnf_matrix_test[1,1]>=',cnf_matrix[1,1])
  
  print(disc + 'matrix_accuracy_score :', (cnf_matrix[1,1] + cnf_matrix[0,0])/(cnf_matrix[1,0]+cnf_matrix[1,1]+cnf_matrix[0,1]+cnf_matrix[0,0])*100)
  print(disc + 'matrix_recall_score :', (cnf_matrix[1,1]/(cnf_matrix[1,0] + cnf_matrix[1,1])* 100))
  
  ### 비정상 거래 검출 모델(Fraud Detection System (9)

def rf(X_train, y_train, X_test, y_test, disc):
  rf = RandomForestClassifier()
  rf.fit(X_train, y_train.ravel()) # resample 한 모델
  y_test_pre = rf.predict(X_test)
  
  cnf_matrix_rf = confusion_matrix(y_test, y_test_pre)
  print(disc + 'matrix_accuracy_score :', (cnf_matrix_rf[1,1] + cnf_matrix_rf[0,0])/(cnf_matrix_rf[1,0]+cnf_matrix_rf[1,1]+cnf_matrix_rf[0,1]+cnf_matrix_rf[0,0])*100)
  print(disc + 'matrix_recall_score :', (cnf_matrix_rf[1,1]/(cnf_matrix_rf[1,0] + cnf_matrix_rf[1,1])* 100))
  
  ### 비정상 거래 검출 모델(Fraud Detection System (10)

if __name__ == '__main__':
#   X_train = creditcard.X_train
#   y_train = creditcard.y_train
#   X_test = creditcard.X_test
#   y_test = creditcard.y_test
  
#   X_smote = creditcard.X_train_res
#   y_smpte = creditcard.y_train_res
  X_smote = X_train_res
  y_smote = y_train_res
  
  logisticR(X_train, y_train, X_test, y_test, 'smote전 + logisticR') # smote 전
  logisticR(X_smote, y_smote, X_test, y_test, 'smote후 + logisticR') # smote 후
  rf(X_train, y_train, X_test, y_test, 'smote전 + RF') # smote 후
  rf(X_smote, y_smote, X_test, y_test, 'smote후 + RF') # smote 후
  
  ### 여기 위까지가 어제꺼
  
  ## Cross Validation (1)

#인위적인 데이터셋을 만듦

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_test_split_():
  #인위적인 테스트셋 만듦
  x, y = make_blobs(random_state = 0)
  
  #데이터와 타겟 레이블을 훈련 세트와 테스트 셋트로 나눔
  x_train ,x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
  
  #모델 학습
  logreg = LogisticRegression().fit(x_train , y_train)
  
  #테스트셋 평가
  print('테스트셋 점수 : {:.2f}'.format(logreg.score(x_test,y_test)))
  
  ## Cross Validation (2)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def k_fold():
  iris = load_iris()
  kf_data = iris.data
  kf_label = iris.target
  kf_columns = iris.feature_names
  
  rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 2019)
  scores = cross_val_score(rf, kf_data, kf_label, cv = 10)
  
  print(scores)
  print('rf k-fold CV score : {:.2f}%'.format(scores.mean()))
  
  ## Cross Validation (3)

from IPython.display import display
import pandas as pd

def k_fold_validate():
  from sklearn.model_selection import cross_validate
  iris = load_iris()
  kf_data = iris.data
  kf_label = iris.target
  kf_columns = iris.feature_names
  
  rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 2019)
  scores = cross_validate(rf, kf_data, kf_label, cv = 10, return_train_score = True)
  
  print('<<score>>')
  display(scores)
  res_df = pd.DataFrame(scores)
  print('<< res_df >>')
  display(res_df)
  print('평균 시간과 점수 : \n', res_df.mean())
  
if __name__ == '__main__':
    train_test_split_()
#   k_fold_cross_val_score()
    k_fold()
    k_fold_validate()
  
  ## Cross Validation (4) - KFold 예제

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
kf_data = iris.keys()
print(' << kf_data >>')
print(kf_data)

kf_data = iris.data
kf_label = iris.target
kf_columns = iris.feature_names

# all+shift+E
kf_data = pd.DataFrame(kf_data, columns = kf_columns)
print('<< kf_label >>')
print(kf_label)
print(pd.value_counts(kf_label))
print(kf_label.sum())
print(kf_label.dtype)

## Cross Validation (5) - KFold 예제

def Kfold():
  from sklearn.model_selection import KFold
  kf = KFold(n_splits = 5, random_state = 0)
  
  #split()는 학습용과 검증용의 데이터 인덱스 출력
  for i, (train_idx, valid_idx) in enumerate(kf.split(kf_data.values, kf_label)):
    
    train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
    valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]
    
    print('{} Fold train label\n{}'.format(i, train_label))
    print('{} Fold valid label\n{}'.format(i, valid_label))
    #print("{} Fold, train_idx{}\n train label\n".format(i, train_idx, train_label))
	
## Cross Validation (6) - KFold 예제

def Stratified_KFold():
  from sklearn.model_selection import StratifiedKFold
  kf = StratifiedKFold(n_splits = 5, random_state = 0)
  
  # Split()는 학습용과 검증용의 데이터 인덱스 출력
  for i, (train_idx, valid_idx) in enumerate(kf.split(kf_data.values, kf_label)):
    
    train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
    valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]
    
    print('{} Fold train label\n{}'.format(i, train_label))
    print('{} Fold valid label\n{}'.format(i, valid_label))
    #print("{} Fold, train_idx{}\n train label\n".format(i, train_idx, train_label))
	
## Cross Validation (7) - KFold 예제

if __name__ == '__main__':
  Kfold()
  Stratified_KFold()
  
## Cross Validation (8) - KFold 예제

def Stratified_KFold_ex():
  from sklearn.model_selection import KFold
  from sklearn.model_selection import StratifiedKFold
  
  kfold = KFold(n_splits = 5)
  stratifiedKfold = StratifiedKFold(n_splits = 5)
  
  iris = load_iris()
  x, y = make_blobs(random_state=0)
  
  # 데이터와 타겟 레이블을 훈련세트와 테스트 셋트로 나눔
  x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
  
  #모델 학습
  logreg = LogisticRegression().fit(x_train, y_train)
  print('교차 검증 점수(kfold) :\n', cross_val_score(logreg, iris.data, iris.target, cv = kfold))
  print('교차 검증 점수(stratifiedkfold) :\n', cross_val_score(logreg, iris.data, iris.target, cv = stratifiedKfold))
  
  ## Cross Validation (9) - KFold 예제

if __name__ == '__main__':
  Kfold()
  Stratified_KFold()
  Stratified_KFold_ex()
  
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
lr = LogisticRegression()

parameters = {'C':np.linspace(1, 10, 1), #숫자는 임의로 정하면 됨
             'penalty':['l1','l2']}

clf = GridSearchCV(lr, parameters, cv = 5, n_jobs = 3) #n_jobs 멀티 프로세스를 사용
print("<< clt - fit >>")
clf.fit(X_smote, y_smote.ravel())
print("<< Best params >>", clf.best_params_, clf.best_estimator_, clf.best_score_)