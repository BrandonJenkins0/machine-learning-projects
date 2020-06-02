"""
This modules contains utility functions to clean up the project flow
"""

# Importing modules
import numpy as np
import matplotlib.pyplot as plt


# Loading data
def load_img_data():
    data = np.load("data/melanoma_data.npz", allow_pickle=True)
    X_train, y_train, X_test = [data[name] for name in data.files]
    return X_train, y_train, X_test


# Plotting images
def plot_images(images, labels, nrows, ncols, figsize=(40, 40)):
    fig, ax = plt.subplots(nrows, ncols, figsize = figsize, constrained_layout=True)
    ax = ax.flatten()
    for num in range(len(ax)):
        ax[num].imshow(images[num])
        ax[num].set_title(labels[num], fontsize=40)
        ax[num].axis('Off')
    plt.show()