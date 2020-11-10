import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# hipertensiDataset = pd.read_csv('iris.csv', names=['umur', 'kegemukan', 'class'], header=0)
data = pd.read_csv('iris.csv')
# fitur = data.iloc[:, 0:2].values
# label = data.iloc[:, -1].values
fitur = data.iloc[:, 0:4].values
label = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(fitur, label, test_size = 1/5, random_state = 0)
print("Data training:")
print(X_train)
print("Data test:")
print(X_test)

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, y_train)

akurasi = model.score(X_train, y_train)
print("Akurasi dari model adlah : {}".format(akurasi))
