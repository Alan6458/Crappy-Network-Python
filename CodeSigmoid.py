from CrappyNetwork import DNN
import matplotlib.pyplot as plt
import numpy as np
import time

with open('images.txt', "r") as f:
    data = [line.rstrip() for line in f]
data = [list(np.dot([int(i[j*3:j*3+3]) for j in range(round(len(i)/3))], 1/256)) for i in data]
for k, a in enumerate(data):
    pixels = [[a[i*28+j] for j in range(28)] for i in range(28)]
    xMin, xMax, yMin, yMax = 27, 0, 27, 0
    for i in range(28):
        if max(pixels[i]) > 0:
            yMin = min(yMin, i)
            yMax = i
            for j in range(28):
                if pixels[i][j] > 0:
                    xMax = max(xMax, j)
                    xMin = min(xMin, j)
    imageRangeX, imageRangeY = xMax-xMin, yMax-yMin
    startX, startY = round(14-imageRangeX/2), round(14-imageRangeY/2)
    moveX, moveY = xMin - startX, yMin - startY
    pixels1 = pixels.copy()
    if moveY > 0:
        pixels1 = pixels[moveY:]
        while len(pixels1) < 28:
            pixels1.append([0 for i in range(28)])
    if moveY < 0:
        pixels1 = pixels[:28+moveY]
        while len(pixels1) < 28:
            pixels1.insert(0, [0 for i in range(28)])
    if moveX > 0:
        pixels1 = [pixels1[i][moveX:] for i in range(28)]
        while len(pixels1[0]) < 28:
            pixels1 = [pixels1[i] + [0] for i in range(28)]
    if moveX < 0:
        pixels1 = [pixels1[i][:28+moveX] for i in range(28)]
        while len(pixels1[0]) < 28:
            pixels1 = [[0] + pixels1[i] for i in range(28)]
    pixels1 = [list(i) for i in np.transpose(pixels1)]
    pixelsTogether = []
    for i in range(28):
        pixelsTogether = pixelsTogether + pixels1[i]
    data[k] = pixelsTogether
with open('answers.txt', "r") as f:
    labels = [line.rstrip() for line in f]
labels = [int(i) for i in labels[0]]
with open('imagesTest.txt', "r") as f:
    dataTest = [line.rstrip() for line in f]
dataTest = [list(np.dot([int(i[j*3:j*3+3]) for j in range(round(len(i)/3))], 1/256)) for i in dataTest]
for k, a in enumerate(dataTest):
    pixels = [[a[i*28+j] for j in range(28)] for i in range(28)]
    xMin, xMax, yMin, yMax = 27, 0, 27, 0
    for i in range(28):
        if max(pixels[i]) > 0:
            yMin = min(yMin, i)
            yMax = i
            for j in range(28):
                if pixels[i][j] > 0:
                    xMax = max(xMax, j)
                    xMin = min(xMin, j)
    imageRangeX, imageRangeY = xMax-xMin, yMax-yMin
    startX, startY = round(14-imageRangeX/2), round(14-imageRangeY/2)
    moveX, moveY = xMin - startX, yMin - startY
    pixels1 = pixels.copy()
    if moveY > 0:
        pixels1 = pixels[moveY:]
        while len(pixels1) < 28:
            pixels1.append([0 for i in range(28)])
    if moveY < 0:
        pixels1 = pixels[:28+moveY]
        while len(pixels1) < 28:
            pixels1.insert(0, [0 for i in range(28)])
    if moveX > 0:
        pixels1 = [pixels1[i][moveX:] for i in range(28)]
        while len(pixels1[0]) < 28:
            pixels1 = [pixels1[i] + [0] for i in range(28)]
    if moveX < 0:
        pixels1 = [pixels1[i][:28+moveX] for i in range(28)]
        while len(pixels1[0]) < 28:
            pixels1 = [[0] + pixels1[i] for i in range(28)]
    pixels1 = [list(i) for i in np.transpose(pixels1)]
    pixelsTogether = []
    for i in range(28):
        pixelsTogether = pixelsTogether + pixels1[i]
    dataTest[k] = pixelsTogether
with open('answersTest.txt', "r") as f:
    labelsTest = [line.rstrip() for line in f]
labelsTest = [int(i) for i in labelsTest[0]]
print("data done")
nodes = [784, 100, 100, 10]
writeToTxt = input("Write to text file?\n")
learningRate = float(input("Learning Rate?\n"))
print(nodes)
network = DNN(nodes)
miniBatchSize = int(input("MiniBatch Size?\n"))
epochs = int(input("epochs?\n"))
processingTimesMins = []
processingTimeTotal = []
rightWrongData = []
percentCorrect = []
print("processing done")
n = 0
for i in range(10000):
    if DNN.calculateOutputSigmoid(network, dataTest[i]) == labelsTest[i]:
        n += 1
print((n/10000)*100)
percentCorrect.append((n/10000)*100)
timeStart = time.perf_counter()
for a in range(epochs):
    for i in range(round(60000/miniBatchSize)):
        network = network.averageAndSubtract([network.backpropSigmoid(data[j], [1 if k == labels[j] else 0 for k in range(10)], learningRate) for j in range(miniBatchSize*i, miniBatchSize*i+miniBatchSize)])
        network.costFull.append(sum(network.cost)/len(network.cost))
        network.cost = []
        print(a, i, round(network.rightWrong/miniBatchSize*100))
        rightWrongData.append(network.rightWrong/miniBatchSize*100)
        network.rightWrong = 0
    processingTimesMins.append((time.perf_counter()-timeStart)/60)
    processingTimeTotal.append(sum(processingTimesMins))
    n = 0
    for i in range(10000):
        if DNN.calculateOutputSigmoid(network, dataTest[i]) == labelsTest[i]:
            n += 1
    if n > 9600:
        learningRate = 0.5
    print((n/10000)*100)
    percentCorrect.append((n/10000)*100)
    timeStart = time.perf_counter()
if writeToTxt == "yes":
    writtenWeights = []
    for i in network.network[0]:
        for j in i:
            for k in list(j):
                writtenWeights.append(str(k)+"\n")
    writtenBiases = []
    for i in network.network[1]:
        for j in list(i):
           writtenBiases.append(str(j)+"\n")
    with open("weightsandbiases", "w") as f:
        f.writelines(writtenWeights)
        f.writelines(writtenBiases)

with open("x".join([str(i) for i in nodes]) + " " + str(miniBatchSize) + " " + str(learningRate) + " S.txt", "w") as f:
    f.writelines(" ".join([str(i) for i in processingTimesMins]))
    f.write("\n")
    f.writelines(" ".join([str(i) for i in processingTimeTotal]))
    f.write("\n")
    f.writelines(" ".join([str(i) for i in network.costFull]))
    f.write("\n")
    f.writelines(" ".join([str(i) for i in rightWrongData]))
    f.write("\n")
    f.writelines(" ".join([str(i) for i in percentCorrect]))
plt.figure(figsize=(10, 6))
plt.grid()
plt.ylim(0, 100)
plt.xlabel("Epoch")
plt.plot(range(1, epochs+1), processingTimesMins, label="Processing Time (Epoch)")
plt.plot(range(1, epochs+1), processingTimeTotal, label="Processing Time (Total)")
plt.plot(np.dot(miniBatchSize/60000, range(len(network.costFull))), network.costFull, label="Network Cost")
plt.plot(np.dot(miniBatchSize/60000, range(len(rightWrongData))), rightWrongData, label="Percent Correct (Each Mini-Batch)")
plt.plot(range(0, epochs+1), percentCorrect, label="Percent Correct (10,000 test cases)")
plt.title("Network: " + "x".join([str(i) for i in nodes]) + ", Minibatch Size: " + str(miniBatchSize) + ", Learning Rate: " + str(learningRate) + ", Activation: Sigmoid")
plt.legend(loc="upper left")
plt.savefig("x".join([str(i) for i in nodes]) + " " + str(miniBatchSize) + " " + str(learningRate) + " S.svg", format="svg")
plt.show()