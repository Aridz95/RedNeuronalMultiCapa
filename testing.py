from red import Red
import random as r

training_data = {'inputs': [[0,0],[0,1],[1,0],[1,1]], 
                 'target': [[1],[0],[0],[1]]}

nn = Red([2,4,1], 0.2)

print("Salidas sin entrenamiento")
print(nn.predict([0,0]))
print(nn.predict([0,1]))
print(nn.predict([1,0]))
print(nn.predict([1,1]))

for _ in range(1000001):
    index = r.randint(0,3)
    nn.train(training_data['inputs'][index], training_data['target'][index])
    if _ % 100000 == 0:
        print("Salidas con entrenamiento " + str(_ / 100000))
        print(nn.predict([0,0]))
        print(nn.predict([0,1]))
        print(nn.predict([1,0]))
        print(nn.predict([1,1]))

