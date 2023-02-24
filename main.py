import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
import myfunctions as fun
import utilitaire as utl

X_train, Y_train, X_test, Y_test  = utl.load_data()

X_trainRe = X_train.reshape(X_train.shape[0], -1) / X_train.max() #Normaliser X =  (X - min)/(max - min)
X_testRe = X_test.reshape(X_test.shape[0], -1) / X_train.max() #Normaliser X =  (X - min)/(max - min)

print("Dimension X: ",X_trainRe.shape)

print("\nDimension y: ",Y_train.shape)


W, b = fun.artificial_neuron(X_trainRe, Y_train, X_testRe, Y_test)
fun.save_model(W, b)

