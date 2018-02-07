import numpy as np 
import tensorflow as tf 
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization

data = []
with open('training_data.txt') as inputfile:
    for line in inputfile:
        data.append(line.strip().split(' '))
        
words = data[0]
X_train = data[1:]
y_train = []

for i in range(len(X_train)):
    y_train.append(X_train[i][0])
    X_train[i] = X_train[i][1:]

X_train = np.array(X_train).astype(np.int)
y_train = np.array(y_train).astype(np.int)

test = []
with open('test_data.txt') as inputfile:
    for line in inputfile:
        test.append(line.strip().split(' '))

words_train = test[0]
X_test = test[1:]
X_test = np.array(X_test).astype(np.int)

model = Sequential()
model.add(Dense(300, input_shape=(1000,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(300, input_shape=(1000,)))
model.add(Dropout(0.1))
model.add(Dense(100, input_shape=(1000,)))
model.add(Activation('softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

fit = model.fit(X_train, y_train, batch_size = 32, epochs = 3, verbose = 1)
score = model.evaluate(X_train, y_train, verbose = 0)
result = model.predict(X_test)
for i in range(len(result)):
    print(result[i])