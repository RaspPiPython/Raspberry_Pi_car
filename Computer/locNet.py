# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:06:48 2018

@author: tranl
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import load_model
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

class locNet:
    @staticmethod
    def build(width, height, depth):
        # initialze the model with "channel last" input
        model = Sequential()
        inputShape = (height, width, depth) # (5, 5, 80)
        chanDim = -1 # for "channel last" data 
        
        # define layers
        model.add(Conv2D(40, (3, 3), padding='same', input_shape=inputShape
                         ,kernel_regularizer=regularizers.l2(0.01)
                         #,activity_regularizer=regularizers.l1(0.01)
                         ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(40, (3, 3), padding='same'
                         ,kernel_regularizer=regularizers.l2(0.01)
                         #,activity_regularizer=regularizers.l1(0.01)
                         ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(80, (3, 3), padding='same'
                         ,kernel_regularizer=regularizers.l2(0.01)
                         #,activity_regularizer=regularizers.l1(0.01)
                         ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(400))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        model.add(Dense(4))
        #model.add(Activation('softmax'))
        
        return model
    
def main():
    # Loading signNet
    signModel = load_model('signNet0.hdf5')
    
    # get paths to image files
    print('<INFO> Getting image paths...')
    filePaths = CNNFunc.paths(r'Data\localizationRaw')
    print('<INFO> Total number of image paths:', len(filePaths))
    
    # load images
    print('<INFO> Loading images with labels...')
    (data, labels) = CNNFunc.loadLocLabels(filePaths) #label format: (x, y, width, height)
    print('<INFO> Current status of dataset:', type(data), data.shape)
    
    print('<INFO> Processing data with signNet...')
    for image in data:
        winIndex = 0
        windowGrid = np.empty((0, 20, 20, 3), 'uint8')
        windowLocs = np.empty((0, 2), int)
        stopSigns = []
        forwardSigns = []
        modelInput = data.astype('float') / 255.0
        for (x, y, window) in CNNFunc.slideWindow(modelInput, ROI=(0, 40, 320, 80), windowSize=(20,20), stepSize=(20,20)):
            window = np.expand_dims(window, axis = 0)
            windowGrid = np.vstack((windowGrid, window))
            windowLocs = np.vstack((windowLocs, (x, y)))
        predictionResults = signModel.predict(windowGrid) 
        for result in predictionResults:
            resultArg = result.argmax()
            if result[resultArg] > 0.8:
                (x, y) = windowLocs[winIndex]
                if resultArg == 1:
                    image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,0,255),1)
                    stopSigns.append((x, y))
                elif resultArg == 2:
                    image = cv2.rectangle(image,(x,y),(x+19,y+19),(255,0,0),1)
                    forwardSigns.append((x, y))
            winIndex += 1
        if len(stopSigns>0):
            stopSigns = CNNFunc.removeSingleWin(stopSigns)
            x1, y1 = (np.min(stopSigns[:, 0]), np.min(stopSigns[:, 1]))
            x2, y2 = (np.max(stopSigns[:, 0]), np.max(stopSigns[:, 1])) 
            
        forwardSigns = CNNFunc.removeSingleWin(forwardSigns)
        
    
    '''
    # one hot encoding the labels
    print('<INFO> Converting labels...')
    labels = CNNFunc.myFitTransform(labels)
    
    #print('<INFO> Normalizing dataset')
    #data = CNNFunc.normalizeTo(data, 0.5)
    
    print('<INFO> Splitting sataset...')
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=40)
    
    # initialize model parameters
    print('<INFO> Compiling model...')
    epochNum = 25
    learningRate = 0.001
    #decayRate = learningRate/epochNum
    #learningMomentum = 0.9
    storingLocation = 'stopLocNet0.hdf5'
    #opt = SGD(lr=learningRate, decay = decayRate, momentum = learningMomentum, nesterov = True)
    opt = SGD(lr = learningRate)
    model = locNet.build()
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    
    # training the network
    print('<INFO> Training network...')
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=1024, epochs=epochNum, verbose=1)
    
    print('<INFO> Evaluating network...')
    predictions = model.predict(testX, batch_size=256)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=['background', 'stopSign', 'forwardSign']))
    
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
    
if __name__ == '__main__': main()