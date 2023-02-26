import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles 
import myfunctions as fun
from utilitaire import load_data

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape(1, y.shape[0])

print("Dimension X: ",X.shape)

print("\nDimension y: ",y.shape)

plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
plt.show()

params = fun.neural_network(X, y, n1=2)
#fun.save_model(params)

