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
