# Importing packages and modules
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Getting the data pulled together
iris = load_iris()
x = iris.data
y = iris.target
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, random_state=22, test_size=.3)

# My k nearest neighbors algorithm


class KNNbj:

    def __init__(self, k=7):
        self.k = k
        self.x_train = np.empty([0,0])
        self.x_test = np.empty([0,0])
        self.y_train = np.empty([0,0])

    def fit(self, x_train, y_train): # make the default value of K sqrt of dataset size
        self.x_train = x_train
        self.y_train = y_train


    def predict(self, x_test):
        self.x_test = x_test
        results = []

        for row in range(len(self.x_test)):

            distances = []
            most_similar = []

            for i in range(len(self.x_train)):
                distances.append([np.sum(np.square(self.x_test[row,:] - self.x_train[i,:])), i])
                distances = sorted(distances)

            for i in range(self.k):
                index = distances[i][1]
                most_similar.append(self.y_train[index])

            results.append(Counter(most_similar).most_common(1)[0][0])

        return np.array(results)


# Fitting and predicting with my clf
my_knn = KNNbj(k=10)
my_knn.fit(x_train2, y_train2)
results_my = my_knn.predict(x_test2) # predicting 45 flowers' class

# Predicting one flower
target = np.array([[4.8, 3.8, 5.2, 1.9]])
my_knn.predict(target)

# Fitting and predicting with sklearn clf
knn_clf = KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(x_train2, y_train2)
results_knn = knn_clf.predict(x_test2)

# Computing accuracies and similarities of the models
similarity = round(accuracy_score(results_my, results_knn), 4) * 100
my_accuracy = round(accuracy_score(y_test2, results_my), 4) * 100
sklearn_accuracy = round(accuracy_score(y_test2, results_knn), 4) * 100
print('The algorithms predictions were {}% similar.\nMy models accuracy: {}%\nSklearn accuracy: {}%'.format(similarity,
                                                                                                         my_accuracy,
                                                                                                         sklearn_accuracy))

# Does my algorithm work with other datasets?
knn_example_dat = pd.read_csv('/Users/brandonjenkins/Desktop/knn_example.csv')
x_example = np.array(knn_example_dat.iloc[:,:3])
y_example = np.array(knn_example_dat.iloc[:,3])
x_train3, x_test3, y_train3, y_test3 = train_test_split(x_example, y_example, random_state=1)
knn_healthy = KNNbj(k=3)
knn_healthy.fit(x_train3, y_train3)
knn_healthy.predict(np.array([[3.7,2.8,4]])) # my model predicted sick
