# Importing necessary modules
from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score

# Loading in data
data = np.load("Fashion-MNIST/data/fashion_mnist_scaled.npz", allow_pickle=True)
X_train, y_train, X_test, y_test = [data[file] for file in data.files]

## Create a very simple nn architecture
filters1 = 10
kernel_size1 = (3, 3)
padding = 'same'

model1 = Sequential([
    Conv2D(filters1, kernel_size1, padding=padding, activation='relu', input_shape=(28, 28, 1)),
    Dropout(.5),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compiling the simple model
model1.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
es = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)]
history = model1.fit(X_train, y_train, validation_split=.1, epochs=100, callbacks=es, verbose=2)

## Creating another simple cnn with more filters!
filters2 = 30
kernel_size1 = (3, 3)
padding = 'same'

model2 = Sequential([
    Conv2D(filters2, kernel_size1, padding=padding, activation='relu', input_shape=(28, 28, 1)),
    Dropout(.5),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compiling the simple model
model2.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
history = model2.fit(X_train, y_train, validation_split=.1, epochs=100, callbacks=es, verbose=2)

# Not much better there...

## Lets try another dense layer before output layer.
model3 = Sequential([
    Conv2D(filters1, kernel_size1, padding=padding, activation='relu', input_shape=(28, 28, 1)),
    Dropout(.5),
    MaxPooling2D(),
    Flatten(),
    Dense(200, activation='relu'),
    Dense(10, activation='softmax')
])

# Compiling the simple model
model3.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
history = model3.fit(X_train, y_train, validation_split=.1, epochs=100, callbacks=es, verbose=2)

# Again not much better...

## Let's go deeper!
## Lets try another dense layer before output layer.
filters4 = 15
model4 = Sequential([
    Conv2D(filters2, kernel_size1, padding=padding, activation='relu', input_shape=(28, 28, 1)),
    Dropout(.5),
    MaxPooling2D(),

    Conv2D(filters4, kernel_size1, padding=padding, activation='relu'),
    Dropout(.5),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

# Check out how many parameters I am training
model4.summary()

# Compiling the simple model
model4.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
history = model4.fit(X_train, y_train, validation_split=.1, epochs=100, callbacks=es, verbose=2)

## Lets try another simple cnn with quite a few filters.
model5 = Sequential([
    Conv2D(50, kernel_size1, padding=padding, activation='relu', input_shape=(28, 28, 1)),
    Dropout(.5),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compiling the simple model
model5.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
history = model5.fit(X_train, y_train, validation_split=.1, epochs=100, callbacks=es, verbose=2)

## Going super wild with it!
## Lets try pretty deep model
model6 = Sequential([
    Conv2D(50, kernel_size1, padding=padding, activation='relu', input_shape=(28, 28, 1)),
    Dropout(.5),
    MaxPooling2D(),

    Conv2D(40, kernel_size1, padding=padding, activation='relu'),
    Dropout(.5),

    Conv2D(20, kernel_size1, padding=padding, activation='relu'),
    Dropout(.5),

    Conv2D(10, kernel_size1, padding=padding, activation='relu'),
    Dropout(.5),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compiling the simple model
model6.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
history = model6.fit(X_train, y_train, validation_split=.1, epochs=100, callbacks=es, verbose=2)

## Going with the simplest best model.

## Predicting with model2. It had the best results with one of the simplest architectures.
predictions = model2.predict_classes(X_test)

# Accuracy 90.52%
accuracy_score(y_test, predictions)

# Saving out best model
model2.save("Fashion-MNIST/data/simple_cnn.h5")