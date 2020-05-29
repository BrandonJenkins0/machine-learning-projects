"""
This module contains functions that I used during the project. Purpose in putting them here is to clean up
my scripts and make it more readable.
"""

# Importing necessary modules
import numpy as np
import matplotlib.pyplot as plt


# Loading in data
def load_data():
    data = np.load("FashionMNIST/data/fashion_mnist_scaled.npz", allow_pickle=True)
    X_train, y_train, X_test, y_test = [data[file] for file in data.files]
    return X_train, y_train, X_test, y_test


# Function for plotting ten images in a small multiples plot
def plot_10_imgs(X, y):
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(28, 28))
    ax = ax.flatten()
    for num in range(len(ax)):
        ax[num].imshow(np.squeeze(X[num]), cmap='Greys', interpolation='nearest')
        plt.title(f"{y[num]}")
        ax[num].axis('Off')
    plt.tight_layout()
    plt.axis('Off')
    plt.show()


# Getting an augmented dataset
def augmenting_data(gen, X, y, num_augmented_imgs=5):
    aug_iter = gen.flow(X, y, batch_size=len(X))
    data = [next(aug_iter) for num in range(num_augmented_imgs)]
    y1 = [data[num][1] for num in range(num_augmented_imgs)]
    x = [data[num][0] for num in range(num_augmented_imgs)]
    return np.concatenate([np.vstack(x), X]), np.concatenate([np.vstack(y1).reshape(-1), y])