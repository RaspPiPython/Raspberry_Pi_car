# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:38:33 2018

@author: tranl
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K
from keras import regularizers
from SupFunctions import CNNFunc

class whitelanesNet0:
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
    
class whitelanesNet1: #accuracy 36/42
    # batch size = 32, epoch = 100, has decay and nesterov
    # learningRate = 
    @staticmethod
    def build(width, height, depth):
        # initialize the model along with the input shape 
        # input is expected to be grayscale image so no need for channels
        model = Sequential()
        inputShape = (height, width, depth)
        
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=inputShape,
                         kernel_regularizer=regularizers.l2(0.01),
                         #activity_regularizer=regularizers.l1(0.01)
                         ))      
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))        
        model.add(Flatten())
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dense(1))
        
        
        return model
    
    
class whitelanesNet2: #accuracy 39/42
    @staticmethod
    def build(width, height, depth):
        # initialize the model along with the input shape 
        # input is expected to be grayscale image so no need for channels
        model = Sequential()
        inputShape = (height, width, depth)
        
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=inputShape,
                         kernel_regularizer=regularizers.l2(0.01),
                         #activity_regularizer=regularizers.l1(0.01)
                         ))      
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))    
        
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape,
                         kernel_regularizer=regularizers.l2(0.01),
                         #activity_regularizer=regularizers.l1(0.01)
                         ))      
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))         
        model.add(Flatten())  
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.25))

        model.add(Dense(1))
        
        
        return model
    
class whitelanesNet3: #37/42
    @staticmethod
    def build(width, height, depth):
        # initialize the model along with the input shape 
        # input is expected to be grayscale image so no need for channels
        model = Sequential()
        inputShape = (height, width, depth)
        
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=inputShape,
                         kernel_regularizer=regularizers.l2(0.01),
                         #activity_regularizer=regularizers.l1(0.01)
                         ))      
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))        
        model.add(Flatten())  
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.25))

        model.add(Dense(1))
        
        
        return model
    
class whitelanesNet: #testing
    @staticmethod
    def build(width, height, depth):
        # initialize the model along with the input shape 
        # input is expected to be grayscale image so no need for channels
        model = Sequential()
        inputShape = (height, width, depth)
        
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=inputShape,
                         kernel_regularizer=regularizers.l2(0.01),
                         #activity_regularizer=regularizers.l1(0.01)
                         ))      
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))        
        model.add(Flatten())  
        
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.25))

        model.add(Dense(1))
        
        
        return model
    
def main():    
    '''from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)'''

    # get paths to image files
    print('<INFO> Getting image paths...')
    filePaths = CNNFunc.paths('F:\FH_Frankfurt\Python Adrian\Working Codes\Test_RegressCNN\Data0')
    print('<INFO> Total number of image paths:', len(filePaths))
    
    # load images
    print('<INFO> Loading images with labels...')
    (data, labels) = CNNFunc.loadProccessedLabels(filePaths)
    data = np.expand_dims(data, axis=3)
    print('<INFO> Current status of dataset:', type(data), data.shape)
    
    print('<INFO> Changing dataset value range to [0, 1]...')
    data = CNNFunc.rangeChange(data, 255.0)
    
    print('<INFO> Splitting sataset...')
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=45)
    
    print('<INFO> Converting labels to float type...')
    labels = labels.astype(dtype='float')
    
    # initialize model parameters
    print('<INFO> Compiling model...')
    epochNum = 100
    learningRate = 0.017
    decayRate = learningRate/epochNum
    learningMomentum = 0.9
    storingLocation = 'regressWlanes4.hdf5'
    opt = SGD(lr=learningRate, decay = decayRate, momentum = learningMomentum, nesterov = True)
    #opt = SGD(lr = learningRate)
    model = whitelanesNet.build(width=32, height=32, depth=1)
    model.compile(loss='mean_squared_error', optimizer=opt,
                  metrics=['accuracy'])
    
    # training the network
    print('<INFO> Training network...')
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=32, epochs=epochNum, verbose=1)
    
    print('<INFO> Evaluating network...')
    timeBegin = time.time()
    predictions = model.predict(testX, batch_size=32)
    timeElapsed = time.time() - timeBegin
    #print(classification_report(testY, predictions,
    #                            target_names=['Reg_output']))
    acc = 0
    for predIndex in range(0, len(predictions)):
        print('Prediction: {} - Ground truth: {}'.format(predictions[predIndex], testY[predIndex]))
        difference = predictions[predIndex]-testY[predIndex].astype(dtype='float')
        if difference < 0.1 :
            acc += 1
    print('Accuracy rate is {}/ {}, or {}'.format(acc, len(predictions), acc/len(predictions)))
    print('Time elapsed is {}s, so it takes {}s for each frame'.format(timeElapsed, timeElapsed/len(predictions)))
    
    print('<INFO> Saving network at {}...'.format(storingLocation))
    model.save(storingLocation)
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochNum), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochNum), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochNum), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochNum), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    '''    
    #model = load_model('trafficSign0.hdf5')
    filePaths = CNNFunc.paths('F:\FH_Frankfurt\Python Adrian\Working Codes\Test_RegressCNN\Data0')
    i = 0
    timeBegin = time.time()
    (data, labels) = CNNFunc.loadProccessedLabels(filePaths)
    #labels = LabelBinarizer().fit_transform(labels)
    timeBegin = time.time()
    prediction = model.predict(data)
    timeElapsed = time.time() - timeBegin
    for filePath in filePaths:        
        print(i, 'Prediction is:', prediction[i], 'Ground truth is:', labels[i]) 
        i += 1
    timeElapsed = time.time() - timeBegin
    processTime = timeElapsed / len(filePaths)
    print('The model took {} s to finish, so the processing time for each image is {} s.'.format(
            timeElapsed, processTime))
    print('Program completed')
    '''
    
if __name__ == '__main__': main()   