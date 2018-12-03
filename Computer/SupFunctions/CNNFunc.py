# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:45:11 2018

@author: tranl
"""

import cv2
import os
import numpy as np

class preprocessors:
    @staticmethod
    def resizing(inputImage, dimensions = (32, 32)):
        #if len(dimensions) != 2:
        #    print('<WARNING> dimensions of resizing function should have a length of 2')
        width, height = dimensions[0], dimensions[1]
        outputImage = cv2.resize(inputImage, (width, height), interpolation = cv2.INTER_AREA)
        return outputImage
    
    @staticmethod
    def removeTop(inputImage, croprate):
        height = inputImage.shape[0]
        lowerHeight = int(height*croprate)
        outputImage = inputImage[lowerHeight:height]
        return outputImage
    
    @staticmethod
    def gray(inputImage, flag = cv2.COLOR_BGR2GRAY):
        outputImage = cv2.cvtColor(inputImage, flag)
        return outputImage
    
    @staticmethod
    def grayMtp(inputImages, flag = cv2.COLOR_BGR2GRAY):
        outputImages = []
        for inputImage in inputImages:
            outputImage = cv2.cvtColor(inputImage, flag)
            outputImages.append(outputImage)
        return np.array(outputImages)
    
    @staticmethod
    # input is grayscale image, output is binary image
    #def extractWhite(inputImage): 
    def extractWhite(inputImage, threshold = (210, 255)): 
        # extract white parts
        #maxVal = np.max(inputImage)
        #if maxVal > 225:
        #    threshold = (maxVal * 9 // 10, maxVal)
        ret, outputImage = cv2.threshold(inputImage, threshold[0],  
                                         threshold[1], cv2.THRESH_BINARY)
        # closing-opening to reduce noise
        #kernel0 = np.ones((1,1),np.uint8)
        kernel1 = np.ones((3,3),np.uint8) 
        #kernel2 = np.ones((5,5),np.uint8)
        outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_CLOSE, kernel1)
        #outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_OPEN, kernel0)
        
        
        return outputImage
    
    @staticmethod
    # change perspective of input image to birdeye view and keep a square in the middle of bottom side
    def carBirdeyeView(inputImage, bottomShrink, heightExpand, squareWidth):
        height, width = inputImage.shape[:2]
        halfSquareWidth = squareWidth//2
        wShrink = int(width*bottomShrink)
        hExpand = int(height*heightExpand)
        pts1 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        pts2 = np.float32([[0, 0], [width-1, 0], 
                           [width-wShrink-1, height+hExpand-1], 
                           [wShrink-1, height+hExpand-1]])
        persMatrix = cv2.getPerspectiveTransform(pts1, pts2)
        outputImage = cv2.warpPerspective(inputImage, persMatrix, (width, height+hExpand))
        outputImage = outputImage[(height+hExpand-squareWidth):(height+hExpand), 
                         width//2-halfSquareWidth:width//2+halfSquareWidth]
        return outputImage

    @staticmethod
    # change perspective of input image to birdeye view and keep a square in the middle of bottom side
    # when resize is applied before birdeye transformation, the image will be distorted a little so
    # this function change the output image region a little to compensate for that
    def carBirdeyeView2(inputImage, bottomShrink, heightExpand, squareWidth):
        height, width = inputImage.shape[:2]
        halfSquareWidth = squareWidth//2
        wShrink = int(width*bottomShrink)
        hExpand = int(height*heightExpand)
        pts1 = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        pts2 = np.float32([[0, 0], [width-1, 0], 
                           [width-wShrink-1, height+hExpand-1], 
                           [wShrink-1, height+hExpand-1]])
        persMatrix = cv2.getPerspectiveTransform(pts1, pts2)
        outputImage = cv2.warpPerspective(inputImage, persMatrix, (width, height+hExpand))
        outputImage = outputImage[(height+hExpand-squareWidth):(height+hExpand), 
                         width//2-halfSquareWidth-1:width//2+halfSquareWidth-1]
        return outputImage
    
    @staticmethod
    # increase brightness of image if it is too dark
    def adjustDark(inputImage):
        maxValue = np.max(inputImage)
        if maxValue < 250:
            gamma = (1.5*(255-maxValue)/255)+1
            invGamma = 1/gamma
            table = np.array([(i/255)**invGamma*255 for i in range(0, 256)]).astype('uint8')
            return cv2.LUT(inputImage, table)
        else:
            return inputImage
        
    @staticmethod
    # turn pixels on both side outside of the track to black
    def removeSides(image, ROI): # (width, height) is the size of the area to be removed
        width, height = ROI
        (imHeight, imWidth) = image.shape[0:2]
        polygon1 = np.array([[0,height-1],[0,0],[width-1,0]], np.int32)
        polygon2 = np.array([[imWidth-1,height-1],[imWidth-1,0],[imWidth-width-1,0]], np.int32)
        cv2.fillPoly(image,[polygon1],(0,0,0))
        cv2.fillPoly(image,[polygon2],(0,0,0))
        return image    

# This paths0 function extract all file paths from the directory        
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

# Attach labels on the images
def loadProcessedLabels(filePaths): # exclusive for model that classify white lanes
    (images, labels) = ([], [])
    for filePath in filePaths:
        image = cv2.imread(filePath, 0) #read img as gray scale img
        label = filePath.split(os.path.sep)[-2] 
        image = preprocessors.resizing(image, (32, 32))
        images.append(image)
        labels.append(label)
    return (np.array(images), np.array(labels))

def loadLabels0(filePaths):
    (images, labels) = ([], [])
    for filePath in filePaths:
        image = cv2.imread(filePath)
        label = filePath.split(os.path.sep)[-2]
        images.append(image)
        labels.append(label)
    return (np.array(images), np.array(labels))

def loadLabels(filePaths, preprocessors = [], gray = False):
    (images, labels) = ([], [])
    for aFilePath in filePaths:
        if gray == False:
            image = cv2.imread(aFilePath)
        elif gray == True:
            image = cv2.imread(aFilePath, -1)
        label = aFilePath.split(os.path.sep)[-2]
        if len(preprocessors) > 0:
            for aPreprocessor in preprocessors:
                image = aPreprocessor(image)
        images.append(image)
        labels.append(label)
    return (np.array(images), np.array(labels))

def loadLocLabels(filePaths):
    (images, labels) = ([], [])
    for filePath in filePaths:
        image = cv2.imread(filePath)
        labelString = filePath.split(os.path.sep)[-1]
        label = np.zeros(4).astype('int')
        for labelIndex in range(0, 4):
            label[labelIndex] = float(labelString[3*labelIndex:(3*labelIndex+3)])
        images.append(image)
        labels.append(label)
    return (np.array(images), np.array(labels))

# I have to write this because LabelBinarizer().fit_transform() does not work when there are only 2 classes
# Transform string labels to arrays of binary values using one hot encoding method
def myFitTransform(labelList):
    classes = ([])
    binarizedList = ([])
    for label in labelList:
        if label in classes:
            pass
        else:
            classes.append(label)
    
    for label in labelList:
        for aClass in classes:
            if label == aClass:
                binarizedLabel = np.zeros(len(classes))
                binarizedLabel[classes.index(aClass)] = 1
                binarizedList.append(binarizedLabel)
                continue
            else:
                pass
           
    binarizedList = np.array(binarizedList)
    binarizedList = binarizedList.astype('int') 
    return binarizedList

#def normalization(data, maxValue):
def rangeChange(data, maxValue):
    output = data.astype('float') / maxValue #should be 255.0 for images
    return output

def normalizeTo(data, normValue):
    output = []
    for image in data:
        image = image * normValue / np.mean(image)
        output.append(image)
    return np.array(output)

# Add dimension if the data consists of gray scale images
def adjustShape(data): 
    # If data consists of grayscale images, add 1 dimension (depth = 1) 
    # since the model expect a 4-dimensional array
    # For example, 200 32x32 grayscale images has a shape of (200, 32, 32)
    # This shape needs to be changed to (200, 32, 32, 1)
    # For channel first backend, this needs to be changed to (200, 1, 32, 32
    shapeLength = len(data.shape)
    if shapeLength == 3:
        output = np.expand_dims(data, axis=3)
    elif shapeLength ==4:
        pass
    else:
        print('<WARNING> The shape of the input is invalid')
    return output

def slideWindow(image, ROI=(0, 60, 320, 60), windowSize=(20,20), stepSize=(5,5)):
    # ROI has the format (x1, y1, width, height) where (x1, y1) is top left corner
    # and (height, width) is the shape of the ROI.
    # windowSize and stepSize has the format (height, width)
    (x1, y1, width, height) = ROI
    (winWidth, winHeight) = windowSize
    (stepHeight, stepWidth) = stepSize
    for y in range(y1, (y1+1+height-winHeight), stepHeight):
        for x in range(x1, (x1+1+width-winWidth), stepWidth):
            yield (x, y, image[y:y+winHeight, x:x+winWidth])
            
def aWindow(image, loc=(0, 0), windowSize=(20, 20)):
    (x, y) = loc
    (height, width) = windowSize
    return image[y:y+height, x:x+width]

def removeSingleWin(potSigns):
    # If the window has no neighbor, it will be removed from the potential list
    for (x, y) in potSigns:
        if (((x-20, y)==potSigns).all(1).any() or ((x, y+20)==potSigns).all(1).any()  
           or ((x+20, y)==potSigns).all(1).any() or ((x, y-20)==potSigns).all(1).any()):
            pass
        else:
            index = np.where(np.all((x, y)==potSigns, axis=1))
            potSigns = np.delete(potSigns, index, axis=0)
    return potSigns
            
def findIndex(directory):
    fileNames = os.listdir(directory)
    index = len(fileNames) + 1
    return index
            
    