import numpy as np
import matplotlib.pyplot as plt
from random import uniform, choice

# https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide
# https://www.3blue1brown.com/lessons/backpropagation-calculus
# add ReLU
# try using lambda for np.vectorize?
sigmoidActivation = lambda a: 1/(1+np.exp(-a))
def sigmoid(a : list[float]):
    return [sigmoidActivation(i) for i in a]
sigmoidV = np.vectorize(sigmoid)
sigmoidLambda = np.vectorize(lambda a: 1/(1+np.exp(-a)))
sigmoidD = lambda a: sigmoidActivation(a)*(1-sigmoidActivation(a))
# return sigmoid(a)*(1-sigmoid(a))
def sigmoidDerivative(a : list[list[float]]):
    return [[sigmoidD(j) for j in i] for i in a]
sigmoidDerivativeV = np.vectorize(sigmoidDerivative)
sigmoidDerivativeLambda = np.vectorize(lambda a: np.exp(-a)/(1+np.exp(-a))**2)

ReLUActivation = lambda a: max(0, a)
def ReLU(a : list[float]):
    return [ReLUActivation(i) for i in a]
ReLUD = lambda a: 0 if a <= 0 else 1
def ReLUDerivative(a : list[list[float]]):
    return [[ReLUD(j) for j in i] for i in a]

class DNN:
    def __init__(self, dimensions : list[int], minMax=[-1, 1]) -> None:
        self.network = [[[[uniform(minMax[0], minMax[1]) for j in range(dimensions[i-1])] for k in range(v)] for i, v in enumerate(dimensions[1:], start=1)], [[uniform(minMax[0], minMax[1]) for j in range(v)] for v in dimensions[1:]]]
        self.cost = []
        self.costFull = []
        self.rightWrong = 0
    def averageAndSubtract(self, subtract):
        self.network = np.subtract(self.network, np.average(subtract, axis=0))
        return self
    def calculateOutputLinear(self, input : list[int]) -> int:
        r = [input]
        for i in range(1, len(self.network[0])+1):
            r.append(list(np.add(np.dot(r[-1], np.flip(np.rot90(self.network[0][i-1], 1), 0)), self.network[1][i-1])))
        return np.argmax(r[-1])
    def calculateOutputSigmoid(self, input : list[int]) -> int:
        r = [input]
        for i in range(1, len(self.network[0])+1):
            r.append(list(sigmoid(np.add(np.dot(r[-1], np.flip(np.rot90(self.network[0][i-1], 1), 0)), self.network[1][i-1]))))
        return np.argmax(r[-1])
    def calculateOutputTanh(self, input : list[int]) -> int:
        r = [input]
        for i in range(1, len(self.network[0])+1):
            r.append(list(np.tanh(np.add(np.dot(r[-1], np.flip(np.rot90(self.network[0][i-1], 1), 0)), self.network[1][i-1]))))
        return np.argmax(r[-1])
    def calculateOutputReLU(self, input : list[int]) -> int:
        r = [input]
        for i in range(1, len(self.network[0])+1):
            r.append(list(ReLU(np.add(np.dot(r[-1], np.flip(np.rot90(self.network[0][i-1], 1), 0)), self.network[1][i-1]))))
        return np.argmax(r[-1])
    def hillClimb(self, goal, dimensions, amount):
        self.network = [[[[goal[0][i-1][k][j]+uniform(-amount, amount) for j in range(dimensions[i-1])] for k in range(v)] for i, v in enumerate(dimensions[1:], start=1)], [[goal[1][i][j]+uniform(-amount, amount) for j in range(v)] for i, v in enumerate(dimensions[1:])]]
        return self
    def updateParamsSigmoid(self, input : list[int], goalOutput : list[int], learningRate : int):
        r = [input]
        rLin = [input]
        for i in range(1, len(self.network[0])+1):
            r.append(sigmoidV(np.add(np.dot(r[-1], np.flip(np.rot90(self.network[0][i-1], 1), 0)), self.network[1][i-1])))
            rLin.append(sigmoidDerivativeV(np.add(np.dot(r[-2], np.flip(np.rot90(self.network[0][i-1], 1), 0)), self.network[1][i-1])))
        error = [np.dot(2, np.subtract(goalOutput, r[-1]))]
        for i in self.network[0][::-1]:
            error.append(np.dot(2, np.dot(error[-1], i)))
        error = error[::-1]
        errorLin = np.dot(learningRate, np.multiply(error, rLin))
        gradient = [np.rot90([np.dot(c, v) for c in r[i]], 1) for i, v in enumerate(errorLin[1:])]
        return [gradient, errorLin[1:]]
    def backpropSigmoid(self, input : list[int], output : list[int], learningRate=1.0):
        # update params: weight_new = weight - learningRate * dC/dweight
        # forward pass
        a = [np.array(input)]
        z = [np.array(input)]
        # transpose is used instead of np.flip(np.rot90(list, 1), 0)
        # z is unactivated weighted sum
        # a is activation
        for i in range(1, len(self.network[0])+1):
            z.append(np.add(np.dot(a[-1], np.transpose(self.network[0][i-1])), self.network[1][i-1]))
            a.append(sigmoid(z[-1]))
        # derivatives go here (later to results)
        backNet = [self.network[0][::-1], self.network[1][::-1]]
        # blank network, multiply derivatives to this (is this necessary?)
        # flips stuff backwards, applies derivative of sigmoid to unactivated sum
        a, z = a[::-1], sigmoidDerivative(z[::-1])
        z[0] = np.multiply(np.multiply(2, np.subtract(a[0], output)), z[0])
        self.cost.append(np.sum(np.power(np.subtract(a[0], output), 2)))
        if np.argmax(a[0]) == np.argmax(output):
            self.rightWrong += 1
        newNetW = []
        newNetB = [z[0]]
        # See notebook, perhaps have biases updated with this now or later?
        # z = [np.multiply(np.multiply(2, np.subtract(a[0], output)), z[0])] + [np.multiply(np.sum(backNet[0][i], axis=0), v) for i, v in enumerate(z[1:])]
        # when I said "It's matrix multiplicationing time!" and matrixly multiplied everywhere
        # loop for weights
        # use insert into newNet
        for i in range(1, len(z)-1):
            z[i] = np.multiply(z[i], np.sum(np.transpose(np.multiply(z[i-1], np.transpose(backNet[0][i-1]))), axis=0))
            newNetB.insert(0, z[i])
            newNetW.insert(0, np.multiply(a[i], np.transpose(np.full((len(a[i]), len(z[i-1])), z[i-1]))))
        newNetW.insert(0, np.multiply(a[-1], np.transpose(np.full((len(a[-1]), len(z[-2])), z[-2]))))
        # order of average changes then add or add every version of changes to their own copy of network and then average does not matter
        return [np.dot(newNetW, learningRate), np.dot(newNetB, learningRate)]
    def backpropReLU(self, input : list[int], output : list[int], learningRate=1.0):
        a = [np.array(input)]
        z = [np.array(input)]
        for i in range(1, len(self.network[0])+1):
            z.append(np.add(np.dot(a[-1], np.transpose(self.network[0][i-1])), self.network[1][i-1]))
            a.append(ReLU(z[-1]))
        backNet = [self.network[0][::-1], self.network[1][::-1]]
        a, z = a[::-1], ReLUDerivative(z[::-1])
        z[0] = np.multiply(np.multiply(2, np.subtract(a[0], output)), z[0])
        self.cost.append(np.sum(np.power(np.subtract(a[0], output), 2)))
        if np.argmax(a[0]) == np.argmax(output):
            self.rightWrong += 1
        newNetW = []
        newNetB = [z[0]]
        for i in range(1, len(z)-1):
            z[i] = np.multiply(z[i], np.sum(np.transpose(np.multiply(z[i-1], np.transpose(backNet[0][i-1]))), axis=0))
            newNetB.insert(0, z[i])
            newNetW.insert(0, np.multiply(a[i], np.transpose(np.full((len(a[i]), len(z[i-1])), z[i-1]))))
        newNetW.insert(0, np.multiply(a[-1], np.transpose(np.full((len(a[-1]), len(z[-2])), z[-2]))))
        return [np.dot(newNetW, learningRate), np.dot(newNetB, learningRate)]