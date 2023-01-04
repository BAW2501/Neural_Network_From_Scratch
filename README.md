# Neural Network implementation from scratch

This code is a simple implementation of a neural network with a single hidden layer using the backpropagation algorithm for training. The network is trained on the MNIST dataset, which consists of 60,000 28x28 grayscale images of handwritten digits and their corresponding labels. The goal of the network is to classify the images into one of 10 classes (0-9).

# Dependencies
`data.py`: This module contains the get_mnist function, which is used to load the MNIST dataset.
`numpy`: This library is used for numerical computing, including operations on arrays and matrices.
`matplotlib`: This library is used to plot the images and their corresponding labels.
# Model Structure
The network consists of:

An input layer with 784 (28x28) nodes, one for each pixel in the image.
A single hidden layer with 20 nodes.
An output layer with 10 nodes, one for each class.
The weights and biases between the layers are initialized randomly using a uniform distribution. The network uses the sigmoid activation function for both the hidden and output layers. The cost function used to measure the error of the network is the mean squared error.

# Training
The network is trained using the backpropagation algorithm, which involves the following steps:

Forward propagation: The input image is passed through the network, and the output is calculated using the current weights and biases.
Cost calculation: The error between the predicted output and the true label is calculated using the cost function.
Backpropagation: The error is backpropagated through the network, and the weights and biases are updated using gradient descent.
Repeat for all images in the training set for a certain number of epochs.
The learning rate and number of epochs can be adjusted to control the speed and convergence of the training process.

# Evaluation
After training, the user can input an index between 0 and 59,999 to see the corresponding image and the predicted label output by the network. The network's accuracy on the training set is also printed after each epoch.

# Usage
To run the code, simply execute `python NN.py`. The user will be prompted to enter an index to see the corresponding image and prediction. The process can be repeated until the user exits the program.