# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:36:43 2018

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

def main():
    # get paths to image files
    print('<INFO> Getting image paths...')
    filePaths = CNNFunc.paths('F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\IsSign')
    print('<INFO> Total number of image paths:', len(filePaths))
    
    # load images
    print('<INFO> Loading images as grayscales with labels...')
    (data, labels) = CNNFunc.loadLabels(filePaths)
    data = CNNFunc.grayMtp(data)
    print('<INFO> Current status of dataset:', type(data), data.shape)
    
    print('<INFO> Changing dataset value range to [0, 1]...')
    data = CNNFunc.rangeChange(data, 255.0)
    
    print('<INFO> Calculating HOG...')
    
    print('<INFO> Training NN...')