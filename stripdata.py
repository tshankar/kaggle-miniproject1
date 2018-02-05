results = []
with open('training_data.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split(' '))
