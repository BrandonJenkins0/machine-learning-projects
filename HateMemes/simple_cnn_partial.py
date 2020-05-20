# Importing modules
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from keras import Sequential
from keras.optimizers import Adam
import numpy as np
import pandas as pd

# How large are the photos
img = load_img("Hate Memes/data/train/hate/01329.png", target_size=(256, 256))
image_arr = img_to_array(img)

# Paths to directories
train_path = "Hate-Memes/data/train/"
val_path = "Hate-Memes/data/val"
test_path = "Hate-Memes/data/"

# Creating data generators
train_batches = ImageDataGenerator().flow_from_directory(train_path, classes=['hate', 'nohate'], batch_size=50)
val_batches = ImageDataGenerator().flow_from_directory(val_path, classes=['hate', 'nohate'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, batch_size=50, class_mode=None, classes=['test'])

# Very simple cnn
simple_mod = Sequential([
    Conv2D(100, (5, 5), padding='same', input_shape=(256, 256, 3), activation='relu'),
    Dropout(.5),
    MaxPool2D(strides=(2, 2)),
    Conv2D(50, (5,5), activation='relu'),
    Dropout(.5),
    MaxPool2D(strides=(2, 2)),
    GlobalAveragePooling2D(),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])

simple_mod.summary()
simple_mod.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the model had to only use some of the train data cause my computer was bout to blow up using all train data
simple_mod.fit_generator(train_batches, steps_per_epoch=20, validation_data=val_batches,
                         validation_steps=50, epochs=10, verbose=2)

# Submitting predictions for this extremely simple model need to clean this up...
predictions = simple_mod.predict_generator(test_batches)
pred_hate = predictions[:, 0]
class_hate = np.where(pred_hate > .5, 1, 0)

# Saving out the model
simple_mod.save('Hate-Memes/data/models/simple_cnn.h5')
