from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from emnist import extract_training_samples, extract_test_samples
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(27, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

images, labels = extract_training_samples('letters')
labels = to_categorical(labels)
images = images.reshape(images.shape[0], 28, 28, 1)

x_val, y_val = extract_test_samples()

model.fit(images, labels, epochs=2, validation_data=(x_val, y_val))
model.save('characterModel')