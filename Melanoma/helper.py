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

# Plotly charts
# feature_list = ['sex','age_approx','anatom_site_general_challenge']
# for  column_name in feature_list:
#     train[column_name].value_counts(normalize=True).to_frame().iplot(kind='bar',
#                                                       yTitle='Percentage',
#                                                       linecolor='black',
#                                                       opacity=0.7,
#                                                       color='red',
#                                                       theme='pearl',
#                                                       bargap=0.8,
#                                                       gridcolor='white',
#                                                       title=f'<b>Distribution of {column_name}