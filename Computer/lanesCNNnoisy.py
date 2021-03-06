# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 20:07:49 2018

@author: tranl
"""
import numpy as np
import matplotlib.pyplot as plt
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

class lanesNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialze the model with "channel last" input
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1 # for "channel last" data 
        #chanDim = 1 # for "channel first" data 
        
        # if we are using "channel first", update the shape
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            
        # define layers
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape
                         ,kernel_regularizer=regularizers.l2(0.01)
                         #,activity_regularizer=regularizers.l1(0.01)
                         ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'
                         ,kernel_regularizer=regularizers.l2(0.01)
                         #,activity_regularizer=regularizers.l1(0.01)
                         ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same'
                         ,kernel_regularizer=regularizers.l2(0.01)
                         #,activity_regularizer=regularizers.l1(0.01)
                         ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'
                         ,kernel_regularizer=regularizers.l2(0.01)
                         #,activity_regularizer=regularizers.l1(0.01)
                         ))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(640))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
    
def main():
    # get paths to image files
    print('<INFO> Getting image paths...')
    filePaths = CNNFunc.paths(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\noisyLanes')
    print('<INFO> Total number of image paths:', len(filePaths))
    
    # load images
    print('<INFO> Loading images with labels...')
    (data, labels) = CNNFunc.loadLabels(filePaths) #data shape is (x, 64, 64, 3)
    data = data[:, :, :, 0] #data are B&W images so only 1 channel is enough
    data = np.expand_dims(data, axis=3) #data shape from (x, 64, 64) to (x, 64, 64, 1)
    print('<INFO> Current status of dataset:', type(data), data.shape)
    
    print('<INFO> Changing dataset value range to [0, 1]...')
    data = CNNFunc.rangeChange(data, 255.0)
    
    # one hot encoding the labels
    print('<INFO> Converting labels...')    
    labels = CNNFunc.myFitTransform(labels)
    
    #print('<INFO> Normalizing dataset')
    #data = CNNFunc.normalizeTo(data, 0.5)
    
    print('<INFO> Splitting sataset...')
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=40)
    
    # initialize model parameters
    print('<INFO> Compiling model...')
    epochNum = 50
    learningRate = 0.0008
    decayRate = learningRate/epochNum
    learningMomentum = 0.9
    storingLocation = 'noisyLanesNet0.hdf5'
    opt = SGD(lr=learningRate, decay = decayRate, momentum = learningMomentum, nesterov = True)
    #opt = SGD(lr = learningRate)
    model = lanesNet.build(width = 64, height = 64, depth = 1, classes = 3)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    
    # training the network
    print('<INFO> Training network...')
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=256, epochs=epochNum, verbose=1)
    
    print('<INFO> Evaluating network...')
    predictions = model.predict(testX, batch_size=256)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=['left', 'right', 'straight']))
    
    print('<INFO> Saving network at {}...'.format(storingLocation))
    model.save(storingLocation)
    print('<INFO> Successfully saved network at {}...'.format(storingLocation))
    
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
    
if __name__ == '__main__': main()