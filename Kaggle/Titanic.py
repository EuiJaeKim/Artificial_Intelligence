import pandas as pd

# 현재 실행 하는 위치 확인 할것 PS C:\Users\xcome\Projects>   < 여기서 실행중.
train = pd.read_csv('./Artificial_Intelligence/Kaggle/Data/Titanic_train.csv',encoding='cp949') # 안읽어지면 인코딩 방식 확인할 것
test = pd.read_csv('./Artificial_Intelligence/Kaggle/Data/Titanic_test.csv',encoding='cp949') 

print(train.head())
print(test.head())
