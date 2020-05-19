# Importing modules
from keras.datasets.fashion_mnist import load_data
import numpy as np

# Loading in the data
(X_train, y_train), (X_test, y_test) = load_data()

# Scale all the data
X_train, X_test = (X_train / 255).reshape(60000, 28, 28, 1), (X_test / 255).reshape(10000, 28, 28, 1)

# Saving out the data to .npz file
np.savez_compressed("Fashion-MNIST/data/fashion_mnist_scaled.npz",
                    X_train = X_train, y_train=y_train, X_test=X_test, y_test=y_test)
