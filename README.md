Good luck getting through the spaghetti code I wrote over the course of 2-3 weeks in February 2023


Decompress the file named "images.zip"

Make sure that Python along with the Numpy, Random, Math, and PyGame libraries are installed


CrappyNetwork.py provides functions for an artificial neural network. This includes forward/back propagation.

Do not use updateParamsSigmoid() because it DOES NOT WORK

backpropReLU() has not been shown to work


images.txt and answers.txt are the 60,000 images of handwritten digits from the MNIST dataset, unpacked from the files from http://yann.lecun.com/exdb/mnist/ using Unpack.py

imagesTest.txt and answersTest.txt are the 10,000 images from the same above link used to test neural networks


CodeSigmoid.py is an example of how the functions in CrappyNetwork.py can be used

CodeSigmoid.py first centers all images loaded, then creates a neural network from the "nodes" variable

The neural network is then trained for a number of epochs and with a minibatch size and learning rate of your choice

You may choose to write the trained network to a file, which can then be read and used by demo.py


demo.py loads a trained neural network (dimensions needs to be specified with the "dimensions" variable on line 13)

demo.py allows you to draw on a screen to test your neural network

Some of the neural networks provided may require you to "transpose" the image before sending it to the calculateOutputSigmoid function using np.transpose(imageSent)
