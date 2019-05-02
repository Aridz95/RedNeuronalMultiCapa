import numpy as np
from random import random
from math import exp

class Red():

    def __init__(self, capas = [2,4,1], learningrate = 0.1):
        if (type(capas) == list and len(capas) > 2):
            self.w = [np.array([[random() for c in range(capas[_])] for r in range(capas[_ + 1])]) for _ in range(len(capas) - 1)]
            self.b = [np.array([[random()] for r in range(capas[_ + 1])]) for _ in range(len(capas) - 1)]

            self.learningrate = learningrate

    def predict(self, inputs):
        if(type(inputs) == list and len(inputs) == self.w[0].shape[1]):
            inp = np.array([[inputs[_]] for _ in range(len(inputs))])
            self.outputs = []
            self.outputs.append(self.activate(np.dot(self.w[0], inp) + self.b[0]))
            for _ in range(1, len(self.w)):
                self.outputs.append(self.activate(np.dot(self.w[_], self.outputs[-1]) + self.b[_]))
            return self.outputs[-1]
    
    def train(self, inputs, target):
        predict = self.predict(inputs)
        tgt = np.array([[target[_]] for _ in range(len(target))])
        error = tgt - predict
        for i in range(1, len(self.w)):
            grad = self.dactivate(self.outputs[-i])
            grad *= (error * self.learningrate)
            delta_w = np.dot(grad, self.outputs[-i-1].T)
            self.w[-i] += delta_w
            self.b[-i] += grad
            error = np.dot(self.w[-i].T, error)


    def activate(self, data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = 1 / (1 + exp(-data[i][j]))
        return data 
    
    def dactivate(self, data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] *= (1 - data[i][j])
        return data 
