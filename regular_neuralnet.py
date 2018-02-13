import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization

results = []
with open('training_data.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split(' '))
        
y_train = []
x_train = []

for i in range(1, len(results)):
    y_train.append(results[i][0])
    x_train.append(results[len(results[i])-1:])
words = results[0]

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train)