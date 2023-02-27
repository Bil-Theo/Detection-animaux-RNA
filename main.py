import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles 
import myfunctions as fun
from utilitaire import load_data

X_train, Y_train, X_test, Y_test = load_data()
X = X.T
y = y.reshape(1, y.shape[0])

print("Dimension X: ",X.shape)

print("\nDimension y: ",y.shape)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.show()

params = fun.neural_network(X, y, n1=2)
#fun.save_model(params)

