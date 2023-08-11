import pygame, random, math
from CrappyNetwork import DNN
import numpy as np
pygame.init()

with open('images.txt', "r") as f:
    data = [line.rstrip() for line in f]
data = [list(np.dot([int(i[j*3:j*3+3]) for j in range(round(len(i)/3))], 1/256)) for i in data]
data = [list(np.transpose([[i[j*28+k] for k in range(28)] for j in range(28)])) for i in data]
data = [[list(j) for j in i] for i in data]
with open("weightsandbiases", "r") as f:
    net = [float(line.rstrip()) for line in f]
dimensions = [784, 100, 100, 10]
networkWeights = [[[0 for j in range(dimensions[i-1])] for k in range(v)] for i, v in enumerate(dimensions[1:], start=1)]
for i, v in enumerate(dimensions[1:], start=1):
    for k in range(v):
        for j in range(dimensions[i-1]):
            networkWeights[i-1][k][j] = net.pop(0)
networkBiases = [[0 for j in range(v)] for v in dimensions[1:]]
for i, v in enumerate(dimensions[1:]):
    for j in range(v):
        networkBiases[i][j] = net.pop(0)

network = DNN([1])
network.network = [networkWeights, networkBiases]
size = width, height = 700, 560
running = True
pixels = [[0 for j in range(28)] for i in range(28)]
pixelsTogether = []
screen = pygame.display.set_mode(size)
mouseDown = False
font = pygame.font.SysFont("Comic Sans MS", 20)
shown = random.randint(0, 59999)
mouseDownPrevious = False
usesRandom = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseDown = True
        if event.type == pygame.MOUSEBUTTONUP:
            mouseDown = False
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(560, 370, 140, 70))
    screen.blit(font.render("Example", False, (0, 0, 0)), (590, 390))
    if 560 < pygame.mouse.get_pos()[0] < 700 and 370 < pygame.mouse.get_pos()[1] < 440 and mouseDown and mouseDown != mouseDownPrevious:
        usesRandom = True
        shown = random.randint(0, 59999)
        pixels = data[shown]
    for i in range(28):
        for j in range(28):
            if math.hypot(pygame.mouse.get_pos()[0]-(i*20+10), pygame.mouse.get_pos()[1]-(j*20+10)) < 30 and mouseDown and pygame.mouse.get_pos()[1] <= 560 and pygame.mouse.get_pos()[0] <= 560:
                pixels[i][j] = max(pixels[i][j], 1-(math.hypot(pygame.mouse.get_pos()[0]-(i*20+10), pygame.mouse.get_pos()[1]-(j*20+10))/30)**8)
                usesRandom = False
            pygame.draw.rect(screen, (255-pixels[i][j]*255, 255-pixels[i][j]*255, 255-pixels[i][j]*255), pygame.Rect(i*20, j*20, 20, 20))
            pygame.draw.rect(screen, (pixels[i][j]*255, pixels[i][j]*255, pixels[i][j]*255), pygame.Rect(i*20+1, j*20+1, 18, 18))
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(560, 140, 140, 70))
    screen.blit(font.render("Clear", False, (0, 0, 0)), (605, 160))
    if 560 < pygame.mouse.get_pos()[0] < 700 and 140 < pygame.mouse.get_pos()[1] < 210 and mouseDown:
        pixels = [[0 for j in range(28)] for i in range(28)]
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
    for i in range(28):
        for j in range(28):
            pygame.draw.rect(screen, (pixels1[i][j]*255, pixels1[i][j]*255, pixels1[i][j]*255), pygame.Rect(i*5+560, j*5, 5, 5))
    # pixels1 = [list(i) for i in np.transpose(pixels1)]
    pixelsTogether = []
    for i in range(28):
        pixelsTogether = pixelsTogether + pixels1[i]
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(560, 220, 140, 70))
    screen.blit(font.render("Prediction: " + str(network.calculateOutputSigmoid(pixelsTogether)), False, (0, 0, 0)), (565, 240))
    pygame.display.flip()
    mouseDownPrevious = mouseDown

pygame.quit()