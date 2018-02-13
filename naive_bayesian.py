from sklearn.naive_bayes import GaussianNB
import numpy as np

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

clf = GaussianNB()
clf.fit(X_train, y_train)
final_result = clf.predict(X_test); 
for i in range(len(final_result)):
    print(final_result[i])