# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:13:23 2018

@author: tranl
"""

from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
import numpy as np
import cv2
import time
import copy
#from SupFunctions import MiscFunc
from SupFunctions import CNNFunc
#from SupFunctions.CNNFunc import preprocessors

def main():
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\left\001.jpg')
    preprocessors = CNNFunc.preprocessors
    procImg = preprocessors.removeTop(image, 3/5)      #cropout unnecessary parts    
    procImg = preprocessors.gray(procImg)                #change to gray scale
    procImg = preprocessors.resizing(procImg, dimensions = (64, 64))
    procImg = preprocessors.extractWhite(procImg)           #extract whitelanes, result is binary image
    print(type(procImg), procImg.shape)
    cv2.imshow('frame', procImg)
    cv2.waitKey(0)

# Test YOLO detection for stopSign
def main6():
    model = load_model('signNet0.hdf5')
    #image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\forwardSignDouble\005.jpg') 
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\localizationRaw\1forward\037.jpg')
    modelInput = CNNFunc.rangeChange(image, 255.0)
    labelList = ['background', 'stopSign', 'forwardSign']    
    counter = 0
    counter2 = 0
    windowGrid = np.empty((0, 20, 20, 3), 'uint8')
    windowLocs = np.empty((0, 2), int)
    winIndex = 0
    potStopSigns = []
    potForwardSigns = []
    timeBegin = time.time()
    for (x, y, window) in CNNFunc.slideWindow(modelInput, ROI=(0, 40, 320, 80), windowSize=(20,20), stepSize=(20,20)):
            counter += 1
            window = np.expand_dims(window, axis = 0)
            windowGrid = np.vstack((windowGrid, window))
            windowLocs = np.vstack((windowLocs, (x, y)))
    #for winIndex in range(0, len(windowLocs)):     
    #    (x, y) = windowLocs[winIndex]
    #    image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,255,0),1)
    '''    
    predictions = model.predict(windowGrid)  
    for aPrediction in predictions:        
        predIndex = aPrediction.argmax()
        if predIndex != 0:
            (x, y) = windowLocs[counter2]
        else:
            counter2 += 1
            continue
        if predIndex == 1:
            image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,0,255),1)            
        elif predIndex == 2:
            image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,255,0),1)        
        print(counter2, 'Raw prediction is:', aPrediction)
        print('Class:', labelList[predIndex], 'Confidence:', '{0:.2f}'.format(aPrediction[predIndex]))
        counter2 += 1
        
    print(type(predictions), predictions.shape)    
    timeElapsed = time.time() - timeBegin
    print('Number of windows is {}, time elapsed is {}s'.format(counter, timeElapsed))
    '''
    
    
    # This part show output from each layer
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    # Testing
    test = windowGrid
    print('Input shape:', windowGrid.shape)
    layer_outs = [func([test, 1.]) for func in functors]
    print(type(layer_outs), len(layer_outs))
    layerCounter = 0 
    #for aLayer_out in layer_outs:        
    #    print(layerCounter, type(aLayer_out), len(aLayer_out), type(aLayer_out[0]), aLayer_out[0].shape)
    #    layerCounter += 1
    
    predictionResults = model.predict(windowGrid)  
        
    # Find rough bounding boxes
    #for result in layer_outs[19][0]:
    for result in predictionResults:
        resultArg = result.argmax()
        if result[resultArg] > 0.8:
            (x, y) = windowLocs[winIndex]
            if resultArg == 1:
                image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,0,255),1)
                potStopSigns.append((x, y))
            elif resultArg == 2:
                image = cv2.rectangle(image,(x,y),(x+19,y+19),(255,0,0),1)
                potForwardSigns.append((x, y))
        winIndex += 1
    print('Stop sign', potStopSigns)
    print('Forward sign', potForwardSigns)
    
    '''
    # Remove potential misclassification
    for aPotStopSign in potStopSigns:
        (x, y) = aPotStopSign
        if (((x-20, y) in potStopSigns) or ((x, y+20) in potStopSigns)  
           or ((x+20, y) in potStopSigns) or ((x, y-20) in potStopSigns)):
            pass
        else:
            potStopSigns.remove(aPotStopSign)
            print(aPotStopSign, 'removed from potStopSigns')
    for aPotForwardSign in potForwardSigns:
        (x, y) = aPotStopSign
        if (((x-20, y) in potForwardSigns) or ((x, y+20) in potForwardSigns)  
           or ((x+20, y) in potForwardSigns) or ((x, y-20) in potForwardSigns)):
            pass
        else:
            potForwardSigns.remove(aPotForwardSign)  
            print(aPotForwardSign, 'removed from potForwardSigns')
    '''        
            
    potStopSigns = np.array(potStopSigns)
    x1, y1 = (np.min(potStopSigns[:, 0]), np.min(potStopSigns[:, 1]))
    x2, y2 = (np.max(potStopSigns[:, 0]), np.max(potStopSigns[:, 1])) 
    if x1>10:
        x1 = x1 - 10
    if x2<290:
        x2 = x2 + 29
    else:
        x2 = x2 + 19  
    y1, y2 = (y1-10, y2+29)  
    image = cv2.rectangle(image,(x1,y1), (x2,y2), (0,255,255), 1)  
    image2 = np.zeros((240, 320, 3), dtype='uint8')
    image2[y1:y2+1, x1:x2+1] = image[y1:y2+1, x1:x2+1]        
    print(x1, y1, x2, y2)   
    print(potStopSigns)
    
    potForwardSigns = np.array(potForwardSigns)
    x1, y1 = (np.min(potForwardSigns[:, 0]), np.min(potForwardSigns[:, 1]))
    x2, y2 = (np.max(potForwardSigns[:, 0]), np.max(potForwardSigns[:, 1]))
    if x1>10:
        x1 = x1 - 10
    if x2<290:
        x2 = x2 + 29
    else:
        x2 = x2 + 19
    y1, y2 = (y1-10, y2+29)        
    image = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,255), 1) 
    image3 = np.zeros((240, 320, 3), dtype='uint8')
    image3[y1:y2+1, x1:x2+1] = image[y1:y2+1, x1:x2+1]       
    print(x1, y1, x2, y2)   
    print(potForwardSigns)
    cv2.imshow('Frame', image)
    cv2.waitKey(0)
    cv2.imshow('Frame', image2)
    cv2.waitKey(0)
    cv2.imshow('Frame', image3)
    cv2.waitKey(0)
            
    '''
    outputImgs = layer_outs[0][0][35]   # output
    outputImgs = (outputImg*255).astype('uint8')
    print(outputImg.shape, outputImg[0])
    '''
    #cv2.imshow('Output', outputImg)
    #cv2.waitKey(0)
    
    
    #cv2.imshow('Image', image)
    #cv2.waitKey(0)
 
           
# Test time for sliding window method with pure CNN
def main5():
    model = load_model('trafficSign1.hdf5')
    #preprocessors = CNNFunc.preprocessors
    #filePaths = CNNFunc.paths('F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\CroppedSigns')    
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\TrafficSigns\speedSign\014.jpg')    
    image1 = copy.copy(image)
    image2 = copy.copy(image)
    image3 = copy.copy(image)
    image4 = copy.copy(image)
    images = (image1, image2, image3, image4)
    imageLabels = ('light', 'others', 'speed', 'stop')
    indexLabel = 0
    image = CNNFunc.rangeChange(image, 255.0)
    #image = np.expand_dims(image, axis = 0)
    #gray = preprocessors.gray(image)
    counter = 0
    timeBegin = time.time()
    if counter == 0:
        for (x, y, window) in CNNFunc.slideWindow(image):
            counter += 1
            #window = window * 0.5 / np.mean(window)
            window = np.expand_dims(window, axis = 0)
            prediction = model.predict(window).argmax()
            if prediction == 0:
                image1 = cv2.rectangle(image1,(x,y),(x+19,y+19),(0,255,0),1)
                pass
            elif prediction == 1:
                image2 = cv2.rectangle(image2,(x,y),(x+19,y+19),(255,255,255),1)
                pass
            elif prediction == 2:
                image3 = cv2.rectangle(image3,(x,y),(x+19,y+19),(255,0,0),1)
                pass
            elif prediction == 3:
                image4 = cv2.rectangle(image4,(x,y),(x+19,y+19),(0,0,255),1)
                pass
    #image = cv2.rectangle(image,(0,0),(20,20),(0,255,0),1)
    timeElapsed = time.time() - timeBegin
    print('Number of windows is {}, time elapsed is {}s'.format(counter, timeElapsed))
    for anImage in images:
        cv2.imshow(imageLabels[indexLabel], anImage)
        cv2.waitKey(0)
        indexLabel += 1


# Test if model produce same result as ground truth
def main4():
    model = load_model('trafficSign0.hdf5')
    filePaths = CNNFunc.paths('F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\CroppedSigns')
    i = 0
    timeBegin = time.time()
    (data, labels) = CNNFunc.loadLabels(filePaths)
    labels = LabelBinarizer().fit_transform(labels)
    prediction = model.predict(data)
    for filePath in filePaths:        
        print(i, 'Prediction is:', prediction[i], 'Ground truth is:', labels[i]) 
        i += 1
    timeElapsed = time.time() - timeBegin
    processTime = timeElapsed / len(filePaths)
    print('The program took {} s to finish, so the processing time for each image is {} s.'.format(
            timeElapsed, processTime))


def main3():
    model = load_model('trafficSign0.hdf5')
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\CroppedSigns\speedSign\Pasted Layer #26.png')
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    print(prediction)


def main2():
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\right\020.jpg')
    preprocessors = CNNFunc.preprocessors
    output = preprocessors.removeTop(image, 11/20)
    output1 = preprocessors.gray(output)
    output2 = preprocessors.extractWhite(output1)
    output3 = preprocessors.carBirdeyeView(output2, 5/12, 9/5, 160)
     
    model = load_model('streetlanes0.hdf5')
    
    processedImage = preprocessors.resizing(output3)
    processedImage = np.expand_dims(processedImage, axis = 0)
    processedImage = np.expand_dims(processedImage, axis = 3)
    #processedImage = np.expand_dims(processedImage, axis = 2)
    
    #preds = model.predict(processedImage, batch_size=32)
    prediction = model.predict(processedImage).argmax()
    if prediction == 0:
        result = 'Left'
    elif prediction == 1:
        result = 'Right'
    elif prediction == 2:
        result = 'Straight'
    else:
        result = '<ERROR> Cannot determine whether result is left, right or straight'
    print('Result is:', prediction, result)
    
    cv2.putText(image, "Label: {}".format(result),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Prediction', image)
    cv2.waitKey()
    '''cv2.imshow('processed', output)
    cv2.waitKey(0)
    cv2.imshow('processed', output1)
    cv2.waitKey(0)
    cv2.imshow('processed', output2)
    cv2.waitKey(0)
    cv2.imshow('processed', output3)
    cv2.waitKey(0)'''

if __name__ == '__main__': main()
