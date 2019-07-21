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
bar_chart('Sex')