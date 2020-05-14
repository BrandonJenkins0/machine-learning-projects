# Importing modules
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Loading in data that was created in the Loading Data.py file
data = np.load("Simple NN - MNIST/data/mnist_scaled.npz", allow_pickle=True)
X_train, X_test, y_train, y_test = [data[f] for f in data.files]
X_train_reshaped = np.expand_dims(X_train.reshape(60000, 28, 28), -1)
X_test_reshaped = np.expand_dims(X_test.reshape(10000, 28, 28), -1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Creating architecture for my CNN model
# 1 Conv layer 1 Pooling Layer 2 Dense Layer
simple_mod = Sequential(
    [Conv2D(10, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
     MaxPool2D((2, 2), strides=(2, 2)),
     Flatten(),
     Dense(50, activation='relu'),
     Dense(10, activation='softmax')
     ]
)

# Compiling the Model
simple_mod.compile(Adam(.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the model an specifying validation set
simple_mod.fit(X_train_reshaped, y_train, batch_size=100, epochs=100, verbose=2, validation_split=.1)

# Making predictions on test set
predictions = simple_mod.predict_classes(X_test_reshaped)
accuracy_score(y_test.astype(int), predictions)

# Loading and getting kaggle data into correct format
kaggle_test = pd.read_csv('Simple NN - MNIST/data/test.csv')
kaggle_test_vals = kaggle_test.values
test = np.expand_dims(kaggle_test_vals.reshape(28000, 28, 28), -1)

# Making predictions on kaggle data and saving
kaggle_predictions = simple_mod.predict_classes(test)
final_csv = pd.DataFrame({'ImageId': range(1,28001), 'Label': kaggle_predictions})
final_csv.to_csv('Simple NN - MNIST/data/kaggle_pred1.csv', index=False)