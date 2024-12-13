import tensorflow as tf
from tensorflow import keras
import gzip

import numpy as np
import matplotlib.pyplot as plt



def load_images(filepath):
    """Loads the image data from the unzipped MNIST image file."""
    with open(filepath, 'rb') as f:
        # Read the first 16 bytes (header for images)
        header = np.fromfile(f, dtype=np.dtype('>i4'), count=4)
        magic, num, rows, cols = header
        
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}. Expected 2051.")
        
        # Now read the image data
        images = np.fromfile(f, dtype=np.uint8)
        print(f"Total number of pixels in data: {images.size}")
        
        # Reshape the image data (num images, rows, cols)
        return images.reshape(num, rows, cols)

def load_labels(filepath):
    """Loads the label data from the unzipped MNIST label file."""
    with open(filepath, 'rb') as f:
        # Read the first 8 bytes (header for labels)
        header = np.fromfile(f, dtype=np.dtype('>i4'), count=2)
        magic, num = header
        
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}. Expected 2049.")
        
        # Now read the label data
        labels = np.fromfile(f, dtype=np.uint8)
        return labels
    
print("hello world")

# using the MNIST Fashion databset.
# 60,000 images for training 
# 10,000 for validation / testing
# pixel data of clothing articles

# loading the dataset
test_images_path = "./t10k-images-idx3-ubyte"
test_labels_path = "./t10k-labels-idx1-ubyte"

train_images_path = "./train-images-idx3-ubyte"
train_labels_path = "./train-labels-idx1-ubyte"

# each of these are type numpy array
train_images = load_images(train_images_path)
train_labels = load_labels(train_labels_path)
test_images = load_images(test_images_path)
test_labels = load_labels(test_labels_path)

# split loaded datat into training and testing

print(test_images.shape)

# picking one pixel to loook at
# each image made up of 28x28 pixels - 0 -225 0 = black 255 = white
print(train_images[0, 23, 23])

# get the first ten labels
print(train_labels[:10])

# class labels, there are 10 mapped to an integer 0 through 9
class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# data preprocessing
# convert all data points to between 0 and 1, making it small to be fed into the neural network - and matching the values to fit the biases within the NN that will be between 0 and 1, if we dont do this the size of the data will overpower the size of the weight
train_images = train_images / 255.0
test_images = test_images / 255.0

# building the model
# Sequential = information going from left side to right side, passing through the layers sequentially
# building the architecture of the neural network
model = keras.Sequential([
    # layer 1 = input layer = 784 neurons, shape 28x28, denotes that images are coming at that shape .Flatten will reshape the (28,28) array into a vector of 784 neurons, so 1 neuron = 1 pixel
    keras.layers.Flatten(input_shape=(28, 28)),
    
    # only one hidden layer. dense means that the layer will be fully connected to each neuron from the previous layer
    # it has 128 neurons - this was chosen randomly
    # activation function used is rectified linear unit
    keras.layers.Dense(128, activation='relu'),
    
    # output layer - Dense layer - 
    # softmax activation function transforms raw outputs from a neural network into a probability distribution. 
    # For each of the 10 class labels, the input will have a probability of belonging to each of these labels
    # softmax activation function will result in each output neuron being 0-1 acting as the probability of it being in that class and the sum of the output neurons being 1
    keras.layers.Dense(10, activation='softmax')
])

# training process
# compiling the model
# defining the optimiser, the loss function, and the metrics of the NN that is being measured
# adam is the function that performs the gradient decent
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# fitting the model to the training data
# this gives us the loss and accuracy against the testing data, the model now must be tested against the test data
model.fit(train_images, train_labels, epochs=10 )


# testing process
# accuracy likely to be lower on the testing data than the training data - this is called overfitting, caused by the number of epochs - the more epochs the more the model gets used to the results of the training data - this is an easy parameter to change
# if the accuracy on the training data is extremely high this can be a property of overfitting
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print("Testing Data Loss: " + str(test_loss))
print("Test Data Accuracy: " + str(test_accuracy))

# the model can now make predictions on an input image's output class label
# predictions = model.predict(test_images)
predictions = model.predict(test_images)
# will return the probability distribution on the input image being a member of each class
print(predictions[0])
# the largest probability will be the prediction for the input images output class label
print(class_names[np.argmax(predictions[0])])

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# can visualise the data to see if the prediction matches the actual result

# verifying predictions
# use the model to make predictions on any entry
# get an image on the test data set make a prediction on that based on the model and show actual result vs prediction
COLOUR = "WHITE"
plt.rcParams["text.color"] = COLOUR

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.ylabel("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
def predict(model, image, correct_label):
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    
    show_image(image, class_names[correct_label], predicted_class)

num = 5786
image = test_images[num]
label = test_labels[num]
predict(model, image, label)