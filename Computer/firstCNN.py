# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:28:17 2018

@author: tranl
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras import regularizers


def paths0(directory):
    fileNames = [f for f in os.listdir(directory)]
    filePaths = []
    for fileName in fileNames:
        filePath = '\\'.join((directory, fileName))
        filePaths.append(filePath)
    return filePaths

# This paths function assume that inside directory, there are folders with names 
# the same as classes and images of each class will be inside their respective folders
def paths(directory):
    classNames = [f for f in os.listdir(directory)]
    classPaths = []
    filePaths = []
    for className in classNames:
        classPath = '\\'.join((directory, className))
        classPaths.append(classPath)
    for classPath in classPaths:
        fileNames = [f for f in os.listdir(classPath)]
        for fileName in fileNames:
            filePath = '\\'.join((classPath, fileName))
            filePaths.append(filePath)
    return filePaths

# include the feature allowing preprocessing here
def loadProccessedLabels(filePaths): 
    (images, labels) = ([], [])
    for filePath in filePaths:
        image = cv2.imread(filePath, 0) #read img as gray scale img
        label = filePath.split(os.path.sep)[-2] 
        image = resizing(image, (32, 32))
        images.append(image)
        labels.append(label)
    return (np.array(images), np.array(labels))

def resizing(image, dimensions): # dimensions is a tuple
    width, height = dimensions[0], dimensions[1]
    resizedImage = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return resizedImage

def normalization(data, maxValue):
    #outputs = []
    #for info in data:
    #    output = info.astype('float') / maxValue
    #    outputs.append(output)
    #return outputs
    output = data.astype('float') / 255.0
    # If data consists of grayscale images, add 1 dimension (depth = 1) 
    # since the model expect a 4-dimensional array
    # For example, 200 32x32 grayscale images has a shape of (200, 32, 32)
    # This shape needs to be changed to (200, 32, 32, 1)
    # For channel first backend, this needs to be changed to (200, 1, 32, 32)
    shapeLength = len(output.shape)
    if shapeLength == 3:
        output = np.expand_dims(output, axis=3)
    return output

class ShallowNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width,)

		# define the first (and only) CONV => RELU layer
		#model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))    
		#model.add(Activation("relu"))
        
		model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape,
                   kernel_regularizer=regularizers.l2(0.01),
                   #activity_regularizer=regularizers.l1(0.01)
                   ))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Flatten())
		#model.add(Dense(32))
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

def main():
    # get paths to image files 
    print('[INFO] getting image paths...')
    pathToFiles = paths('F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Output')    
    print('Total number of image paths:', len(pathToFiles))
    
    # load images 
    print('[INFO] loading images with labels...')
    (data, labels) = loadProccessedLabels(pathToFiles)   # images already resized to 32x32       
    #data = resizing(data, (32, 32))
    #for i in range(0, 121, 60):
    #    print(i, labels[i])
        #print(type(data), len(data))
    #    print(type(data [i]), len(data[i]), data[i].shape)
    #    cv2.imshow('Image {}'.format(i), data[i])
    #    cv2.waitKey(0)  
    
    # scale pixel values to range [0, 1]
    print('[INFO] Changing dataset value range to [0, 1]...')
    #data = normalization(data, 255.0)
    data = data.astype('float') / 255.0
    
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    print('[INFO] splitting dataset...')
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.2, random_state=45)

    # convert the labels from integers to vectors
    # works only when there are more than 2 classes
    # for 2 classes, see https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    print('[INFO] converting labels...')
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)
    
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    epochNum = 100
    learningRate = 0.05
    decayRate = learningRate/epochNum
    learningMomentum = 0.95
    storingLocation = 'streetlanes1.hdf5'
    opt = SGD(lr=learningRate, decay = decayRate, momentum = learningMomentum, nesterov = True)
    model = ShallowNet.build(width=32, height=32, depth=1, classes=3)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=32, epochs=epochNum, verbose=1)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=["left", "right", "straight"]))
    
    print('<INFO> serializing network...')
    model.save(storingLocation)
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
        
    print('Program completed')
    
if __name__ == '__main__': main()