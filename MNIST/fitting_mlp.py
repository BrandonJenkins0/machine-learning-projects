# Importing modules
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Loading data in and creating test and train sets
data = np.load('MNIST/data/mnist_scaled.npz', allow_pickle=True)
X_train, X_test, y_train, y_test = [data[f] for f in data.files]

# Building model architecture
model = Sequential()
model.add(Dense(50, input_shape=(784,), activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiling and fitting the model
model.compile(SGD(lr=.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=100, validation_split=.1, epochs=100, verbose=2)

# Calculating prediction probabilities and prediction classes
predictions = model.predict(X_test, batch_size=50, verbose=0)
rounded_predictions = model.predict_classes(X_test, batch_size=50, verbose=0)

# Creating dataset of predicted class and probability of that class
pred_target = []
prob = []
for i, j in enumerate(rounded_predictions):
    pred_target.append(j)
    prob.append(predictions[i, j])
results = pd.DataFrame({'Predicted': pred_target, 'Confidence': prob})

# Grabbing 25 instances where model was least confident in its prediction
not_confident = results.sort_values('Confidence').head(25)
img = X_test[not_confident.index]
correct_lab = y_test[not_confident.index]
pred = not_confident['Predicted'].values

# Visualizing the instances that the model wasn't that confident in.
fig, ax = plt.subplots(nrows=5, ncols=5)
ax = ax.flatten()

for num in range(25):
    image = img[num].reshape(28, 28)
    ax[num].imshow(image, cmap='Greys', interpolation='nearest')
    ax[num].set_title(f"{num + 1}) t: {correct_lab[num]} p: {pred[num]}")

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Accuracy Score
accuracy_score(y_test.astype(int), rounded_predictions)

# Confusion Matrix
cm = confusion_matrix(y_test.astype(int), pred_target)
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap='Blues')
plt.show()

# Saving out the model
model.save('MNIST/data/first_mnist_model.h5')
