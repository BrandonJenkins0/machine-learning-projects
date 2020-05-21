# Try out some data augmentation to make it better.
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from FashionMNIST.helpers import load_data, plot_10_imgs, augmenting_data
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Loading in data
X_train, y_train, X_test, y_test = load_data()

# Grabbing first image as example and plotting it
ex1 = X_train[0]
y1 = y_train[0]
image1 = np.expand_dims(ex1, 0)
plt.imshow(image1[0,:,:,0], cmap="Greys")
plt.show()

# Creating image generator
gen = ImageDataGenerator(rotation_range=10, width_shift_range=.1, zoom_range=.1,
                         horizontal_flip=True)

# Creating image iterator; will create one augmented photo each per each iteration
aug_iter = gen.flow(image1)

# Get 10 samples of augmented photo and plotting them
aug_img_sample = [next(aug_iter) for num in range(10)]
plot_10_imgs(aug_img_sample, [0]*10)

# Creating augmented dataset
X_train_augmented, y_train_augmented = augmenting_data(gen, X_train, y_train, 2)

# Lets try training same architecture from scratch
## Creating another simple cnn with more filters!
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')
callbacks = [es, mcp_save, reduce_lr_loss]

model1 = Sequential([
    Conv2D(50, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),

    Conv2D(20, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(40, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compiling the simple model
model1.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
model1.fit(X_train_augmented, y_train_augmented, validation_split=.1, epochs=100,
           callbacks=callbacks, verbose=2)

# What kind of accuracy do I have on the test data?
predictions = model1.predict_classes(X_test)
accuracy_score(y_test, predictions)

# One last crazy attempt with the augmented data
model2 = Sequential([
    Conv2D(120, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(),

    Conv2D(60, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(30, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compiling the simple model
model2.compile(Adam(.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting model with specifying validation set
model2.fit(X_train_augmented, y_train_augmented, validation_split=.1, epochs=100,
           callbacks=callbacks, verbose=2)


