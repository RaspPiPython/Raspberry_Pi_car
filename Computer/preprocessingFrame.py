# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:29:47 2018

@author: tranl
"""

import numpy as np
import cv2
import argparse
import time
import random
from PIL import Image
from SupFunctions import CNNFunc

class preProcessors:
    def __init__(self, imageArray):        
        self.imageArray = imageArray
        
    def processor1(self, cropHeight, halfSquareWidth): 
        self.cropHeight = cropHeight #height to remove, can be float   
        self.halfSquareWidth = halfSquareWidth #width of the output square image, must be int
        
        imgHeight, imgWidth = self.imageArray.shape[:2]
        imgHeightLower = int(imgHeight*cropHeight)
        gray = cv2.cvtColor(self.imageArray, cv2.COLOR_BGR2GRAY)
        cropped = gray[imgHeightLower:imgHeight] #remove top 11/20 of image
        
        '''Keep only the white lanes'''
        ret, white_lanes = cv2.threshold(cropped, 200,255, cv2.THRESH_BINARY)
        wlHeight, wlWidth = white_lanes.shape[:2] 
        
        '''Dilation and Erosion to remove noise on the white lanes'''
        kernel = np.ones((5,5),np.uint8)        
        white_lanes = cv2.dilate(white_lanes, kernel,iterations = 1)
        white_lanes = cv2.erode(white_lanes, kernel,iterations = 1)
        
        '''Change perpective to birdeye's view'''
        indentX = wlWidth*5//12
        expansionY = wlHeight*9//5
        pts1 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-1, wlHeight-1], [0, wlHeight-1]])
        pts2 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-indentX-1, 
                           wlHeight+expansionY-1], [indentX-1, wlHeight+expansionY-1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        mapped = cv2.warpPerspective(white_lanes, M, (wlWidth, wlHeight+expansionY))
        cropmap = mapped[(wlHeight+expansionY-halfSquareWidth*2):(wlHeight+expansionY), 
                         wlWidth//2-halfSquareWidth:wlWidth//2+halfSquareWidth]
        return cropmap

def main():
    image = cv2.imread(r'Data\stopSignDouble\001.jpg')
    for (x, y, window) in CNNFunc.slideWindow(image, ROI=(0, 40, 320, 80), 
         windowSize=(20,20), stepSize=(20,20)):
        image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,0,255),1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
   
def main7():
    preprocessors = CNNFunc.preprocessors
    image = cv2.imread(r'Data\left2\left8.png')
    proc1 = preprocessors.removeTop(image, 3/5)      #cropout unnecessary parts    
    proc2 = preprocessors.gray(proc1)                #change to gray scale  
    proc3 = preprocessors.resizing(proc2, dimensions = (64, 64))
    proc4 = preprocessors.extractWhite(proc3)
    images = [image, proc1, proc2, proc3, proc4]
    for img in images:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
    
# Augment the images with noise and save them
# Each class will be augmented with noise 4 times
def main6():
    inputPath = 'Data\\procLeft'
    outputPath = 'Data\\procLeftNoisy\\'
    label = 'lNoisyBlack'
    index = 0
    stopFlag = False
    filePaths = CNNFunc.paths0(inputPath)
    for loopNum in range(0, 4):
        for path in filePaths:
            image = cv2.imread(path)
            imHeight, imWidth = image.shape[0:2]
            noiseWidth = random.randint(0, imHeight//2)
            noiseX = random.randint(0, imHeight-noiseWidth-1)
            noiseY = random.randint(0, imWidth-noiseWidth-1)
            # Add black noise part 1, comment out if white noise is needed
            # This part find regions with sufficient white portion(>=30%)
            if np.mean(image[noiseX:noiseX+noiseWidth, noiseY:noiseY+noiseWidth]) < 30:
                stopFlag = False
                while(stopFlag == False):
                    noiseWidth = random.randint(0, imHeight//2)
                    noiseX = random.randint(0, imHeight-noiseWidth-1)
                    noiseY = random.randint(0, imWidth-noiseWidth-1)
                    if np.mean(image[noiseX:noiseX+noiseWidth, noiseY:noiseY+noiseWidth]) >= 30:
                        stopFlag = True
                        loopNum += 1
                    if loopNum > 10:
                        stopFlag = True # prevent program from looping more than 10 times for each image
            # End part 1
            
            #noiseRate = random.randint(5, 95) # noise rate 5% to 95% for white noise
            noiseRate = random.randint(50, 100) # noise rate 50% to 100% for black noise
            
            noiseArray = np.zeros(noiseWidth*noiseWidth).astype('uint8')
            for i in range(0, len(noiseArray)):
                if random.randint(0, 99) < noiseRate:
                    noiseArray[i] = 255            
            noiseWindow = noiseArray.reshape(noiseWidth, noiseWidth)
            noiseWindow = np.stack((noiseWindow,)*3, -1)            
            # Add black noise part 2, comment out if white noise is needed
            image[noiseX:noiseX+noiseWidth, noiseY:noiseY+noiseWidth] = cv2.subtract(
                 image[noiseX:noiseX+noiseWidth, noiseY:noiseY+noiseWidth], noiseWindow)
            # End part 2   
            
            # Add white noise, comment out if black noise is needed
            #image[noiseX:noiseX+noiseWidth, noiseY:noiseY+noiseWidth] = cv2.add(
            #     image[noiseX:noiseX+noiseWidth, noiseY:noiseY+noiseWidth], noiseWindow)
            # End add white noise
            cv2.imwrite(r'{}{}{:03}.jpg'.format(outputPath, label, index), image)
            index += 1
        print('Finished working at {} and {}, {} noisy pictures created'.format(inputPath, outputPath, index))
        
        
# Augment the images by flipping them 
def main5():
    inputPath = 'Data\\procStraight'
    outputPath = 'Data\\procStraight\\'
    label = 'flippedStraight'
    i = 1
    filePaths = CNNFunc.paths0(inputPath)
    
    for path in filePaths:
        image = cv2.imread(path)
        procImg = cv2.flip(image, 1)
        cv2.imwrite(r'{}{}{:03}.jpg'.format(outputPath, label, i), procImg)     
        i += 1

    
# preprocess and store input images    
def main4():
    preprocessors = CNNFunc.preprocessors
    filePaths = CNNFunc.paths0(r'Data\left2')
    label = 'left'
    outputPath = 'Data\\procLeft\\'
    i = 61  #start point of image names
    
    for path in filePaths:
        image = cv2.imread(path)
        procImg = preprocessors.removeTop(image, 3/5)      #cropout unnecessary parts    
        procImg = preprocessors.gray(procImg)                #change to gray scale
        procImg = preprocessors.resizing(procImg, dimensions = (64, 64))
        procImg = preprocessors.extractWhite(procImg)           #extract whitelanes, result is binary image

        cv2.imwrite(r'{}{}{:03}.jpg'.format(outputPath, label, i), procImg)        
        i += 1
    
    
    '''
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\left\020.jpg')
    image2 = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Output\left\020.jpg')
    procImg = preprocessors.removeTop(image, 3/5)
    procImg = preprocessors.gray(procImg)     
    procImg = preprocessors.resizing(procImg, dimensions = (64, 64))
    procImg = preprocessors.extractWhite(procImg)
    #procImg = preprocessors.resizing(procImg, dimensions = (80, 24))
    print(procImg.shape)
    #procImg2 = cv2.add(procImg//2, procImg) 
    #procImg2 = procImg
    #procImg3 = preprocessors.carBirdeyeView(procImg2, 5/12, 8/5, 60)
    #procImg4 = preprocessors.resizing(procImg3) 
    #print(procImg3.shape)
    #cv2.imwrite(r'Pictures\001.jpg', procImg3)
    
    image2 = preprocessors.resizing(image2)
    
    cv2.imshow('image', procImg)
    cv2.waitKey(0)
    cv2.imshow('image', image2)
    cv2.waitKey(0)
    '''
        
    '''    
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\straight\012.jpg')
    image = preprocessors.removeTop(image, 13/20)
    polygon1 = np.array([[0,39],[0,0],[79,0]], np.int32)
    polygon2 = np.array([[319,39],[319,0],[239,0]], np.int32)
    cv2.fillPoly(image,[polygon1],(0,0,0))
    cv2.fillPoly(image,[polygon2],(0,0,0))
    cv2.imshow('image2', image)
    cv2.waitKey(0)
    image = preprocessors.removeSides(image, (80, 50))
    
    cv2.imshow('image3', image)
    cv2.waitKey(0)
    '''
      
def main3():
    filePaths = CNNFunc.paths0(r'Data\right2')
    print (filePaths)
    preprocessors = CNNFunc.preprocessors
    i=60
    label = 'right'
    
    for path in filePaths:
    #    image = cv2.imread(r'Data\right2\right{}.jpg'.format(i))
        image = cv2.imread(path)        
        i+=1
        croppedImg = preprocessors.removeTop(image, 13/20)      #cropout unnecessary parts    
        procImg = preprocessors.gray(croppedImg)                #change to gray scale
        procImg = preprocessors.extractWhite(procImg)           #extract whitelanes, result is binary image
        procImg = preprocessors.resizing(procImg)               #resize to 32x32 to feed into network
        #procImg2 = preprocessors.carBirdeyeView(procImg, 5/12, 9/5, 160)        
        cv2.imwrite(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_RegressCNN\ProcOutput\right\{}{:03}.jpg'.format(label, i), procImg)
        
        #cv2.imshow('img', image)
        #cv2.waitKey(0)
        #cv2.imshow('img', procImg2)
        #cv2.waitKey(0)
    #    processor = preProcessors(image)
    #    cropmap = processor.processor1(11/20, 80)
    #    cv2.imwrite('Output2\right\{:03}.jpg'.format(i), cropmap)
    

def main2():
    filePaths = CNNFunc.paths0(r'Data\straight2')
    print (filePaths)
    preprocessors = CNNFunc.preprocessors
    i=0
    for path in filePaths:
    #    image = cv2.imread(r'Data\right2\right{}.jpg'.format(i))
        image = cv2.imread(path)        
        i+=1
        procImg = preprocessors.removeTop(image, 11/20)        
        procImg = preprocessors.gray(procImg)
        procImg = preprocessors.extractWhite(procImg)
        procImg2 = preprocessors.carBirdeyeView(procImg, 5/12, 9/5, 160)        
        cv2.imwrite(r'Output2\straight\{:03}.jpg'.format(i), procImg2)
        
        #cv2.imshow('img', image)
        #cv2.waitKey(0)
        #cv2.imshow('img', procImg2)
        #cv2.waitKey(0)
    #    processor = preProcessors(image)
    #    cropmap = processor.processor1(11/20, 80)
    #    cv2.imwrite('Output2\right\{:03}.jpg'.format(i), cropmap)

def main1():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--data", required=True, help="Path to dataset")
    #ap.add_argument("-o", "--output", required=True, help="Path to processed dataset")
    #args = vars(ap.parse_args()) 
    
    timeBegin = time.time()
    
    for i in range(1, 61):
        image = cv2.imread(r'Data\right\{:03}.jpg'.format(i))
        processor = preProcessors(image)
        cropmap = processor.processor1(11/20, 80)
        cv2.imwrite('Output\Right\{:03}.jpg'.format(i), cropmap)
        
    timeElapsed = time.time() - timeBegin
    print('Time elapsed is:', timeElapsed)
    
    

def main0():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-d", "--data", required=True, help="Path to dataset")
    #ap.add_argument("-o", "--output", required=True, help="Path to processed dataset")
    #args = vars(ap.parse_args())    
    
    timeBegin = time.time()
    
    for i in range(1, 2):        
        #image = cv2.imread(args["data"] + "/{}.jpg".format(i))
        i2 = 25
        image = cv2.imread('Data\left\{:03}.jpg'.format(i2))
        imgHeight, imgWidth = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropHeight = imgHeight * 11 // 20 #remove the top 11/20 of image
        #cropWidth = imgWidth * 3 // 16                    
        #cropped = gray[120:320]
        cropped = gray[cropHeight:imgHeight]
        
        '''Keep only the white lanes'''
        ret, white_lanes = cv2.threshold(cropped, 200,255, cv2.THRESH_BINARY)
        #mask_white = cv2.inRange(cropped, 200, 255)
        #white_lanes = cv2.bitwise_and(cropped, mask_white)
        wlHeight, wlWidth = white_lanes.shape[:2] 
        
        '''Dilation and Erosion to remove noise on the white lanes'''
        kernel = np.ones((5,5),np.uint8)        
        white_lanes = cv2.dilate(white_lanes, kernel,iterations = 1)
        white_lanes = cv2.erode(white_lanes, kernel,iterations = 1)    
        #dilation = cv2.dilate(erosion, kernel,iterations = 1)
        #erosion = cv2.erode(dilation, kernel,iterations = 1)  
        
        '''Change perpective to birdeye's view'''
        indentX = wlWidth*5//12
        expansionY = wlHeight*9//5
        pts1 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-1, wlHeight-1], [0, wlHeight-1]])
        pts2 = np.float32([[0, 0], [wlWidth-1, 0], [wlWidth-indentX-1, wlHeight+expansionY-1], [indentX-1, wlHeight+expansionY-1]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        mapped = cv2.warpPerspective(white_lanes, M, (wlWidth, wlHeight+expansionY))
        #cropmap = mapped[cropHeight:(wlHeight+expansionY), cropWidth:250]
        cropmap = mapped[(wlHeight+expansionY-160):(wlHeight+expansionY), wlWidth//2-80:wlWidth//2+80]
        
        '''Print elapsed time (should spend less than 0.1s for each frame)'''
        elapsedTime = time.time()-timeBegin
        print('Elapsed time:', elapsedTime)
        
        '''Show result images'''
        #cv2.imshow('cropped', cropped)
        #cv2.imshow('white_lanes', white_lanes)
        #cv2.waitKey(0)
        #cv2.imshow('mapped', mapped)
        #cv2.waitKey(0)
        cv2.imshow('cropmap', cropmap)
        cv2.waitKey(0)
        
        

if __name__ == '__main__': main()    