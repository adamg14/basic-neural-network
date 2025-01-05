# Neural Network 

## Description
A Python implementation of a neural network with multiple nodes, weights, biases and layers to carry out a classification task.

## Script breakdown
 - [neural-network.py](neural-network.py) - Using MNIST fashion data, which consists of images of items of clothing belonging to a class of type of clothing. The image data is partioned into training and testing data, 60k and 10k images respectively. The pixel image data is pre-processed (through normalisation) to a 28x28 pixel image.
The neural network model is then built with three layers:
  - The first layer flatterns the 28x28 pixel image into a single 1D vector of pixel data, of length 784
  - A hidden dense layer with 128 nodes, which uses the activation function rectified linear unit
  - The output layer, which is a dense layer with 10 nodes to represent the number of possible classes. This layer uses the activation function of softmax which converts the raw          output from the neural network into a probability distribution.
Once the model is build with the architecture outlined above, the model is compiled and then fitted to the training data. The model is then able to make predictions on new image data with a relatively high accuracy.

### Function breakdown
- load_images - Given a filepath to an image, this function returns the shape of the image data.
- load_label - Given an image filepath, this function returns the label of the corresponding image.
- predict - The arguments to this function is: the model, the image and the correct label. This function uses the model trained within the script to make a prediction on the inputted image and compares it with the correct label argument.
- show_image - With the help of matplotlib, this function visualises the image and within the figure shows the user the predicted label and the actual label, to allow the user to compare the results with the actual image.

## Installations/Dependencies
This project uses the Python version 3.10. Therefore, ensure that the python >=3.10 in installed on your machine or create a python virtual environement where the version of python is >= 3.10.

Install the dependencies for this project using the [pip python package manager](https://pypi.org/)

```bash
pip3.10 install tensorflow
pip3.10 install numpy
pip3.10 install matplotlib
```

## Usage 
```bash
python3.10 neural-networks.py
```

## Usage
```bash
python3.10 neural-network.py
```
