import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

data = []
with open('training_data.txt') as inputfile:
    for line in inputfile:
        data.append(line.strip().split(' '))

words_train = data[0]
X_train = data[1:]
y_train = []

for i in range(len(X_train)):
    y_train.append(X_train[i][0])
    X_train[i] = X_train[i][1:]

X_train = np.array(X_train)
y_train = np.array(y_train)

data = []
with open('test_data.txt') as inputfile:
    for line in inputfile:
        data.append(line.strip().split(' '))

words_test = data[0]
X_test = data[1:]
X_test = np.array(X_test)

# Create model given the constraints in the problem
model = Sequential()
model.add(Dense(1000, input_shape=(1000,)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Activation('sigmoid'))
model.add(Dense(200, input_shape=(1000,)))
model.add(Dropout(0.2))
model.add(Activation('sigmoid'))
model.add(Dense(50, input_shape=(1000,)))
model.add(Activation('sigmoid'))

# Predict probabilities
model.add(Dense(1))
model.add(Activation('softmax'))


model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1)
predictions = model.predict(X_test, batch_size=None, verbose=0, steps=None)
print (predictions)

# f = open("submission1.txt", "w+")
# f.write("Id,Prediction\n")
# for i in range(len(predictions)):
#     f.write( str(i+1) + "," + str(int(predictions[i][0])) + "\n")

# Printing the accuracy of our model, according to the loss function specified in model.compile above
score = model.evaluate(X_train, y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
