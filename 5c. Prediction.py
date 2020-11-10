import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 6

data = pd.read_csv('iris.csv', header = 0)
fitur = data.iloc[:, 0:2].values
label = data.iloc[:, -1].values

model = KNeighborsClassifier(n_neighbors, weights='distance')
model.fit(fitur, label)

length = float(input('Masukkan Panjang Sepal (cm) : '))
width = float(input('Masukkan Lebar Sepal (cm) : '))
prediction = model.predict([[length, width]])
print('Prediction : ' + prediction)
