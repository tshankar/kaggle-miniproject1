from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import math 

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

# use cross validation to predict how good it is on the actual training set 
X_train, X_testtraining, y_train, y_testtraining = train_test_split(X_train, y_train, test_size = 0.3, random_state=1)

test = []
with open('test_data.txt') as inputfile:
    for line in inputfile:
        test.append(line.strip().split(' '))

words_train = test[0]
X_test = test[1:]
X_test = np.array(X_test).astype(np.int)

clf = svm.SVC(C = 25, gamma = 0.2)
clf.fit(X_train, y_train)
print(clf.score(X_testtraining, y_testtraining))
final_result = clf.predict(X_test); 
for i in range(len(final_result)):
    print(final_result[i])