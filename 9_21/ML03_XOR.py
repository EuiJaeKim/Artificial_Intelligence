from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier(n_neighbors = 1)

model.fit(x_data,y_data)

x_test = [[0,0],[1,0],[0,1],[1,1]]
y_predict = model.predict(x_test)

print(x_test," : ",y_predict)
print("acc = ",accuracy_score(y_data,y_predict))