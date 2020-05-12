# Importing Modules
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = ((X/255) - .5) * 2
plt.imshow(X[1].reshape(28, 28), cmap='Greys')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=1,
                                                    stratify=y)

np.savez_compressed('MNIST/data/mnist_scaled.npz',
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test)
