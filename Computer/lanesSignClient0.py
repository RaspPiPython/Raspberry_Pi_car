# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:44:43 2018

@author: tranl
"""

from keras.models import load_model
from SupFunctions.ServerClientFunc import PiImageClient
from SupFunctions import CNNFunc
import time
import pickle
import cv2
import numpy as np

def main():
    # Initialization
    ImageClient = PiImageClient()
    preprocessors = CNNFunc.preprocessors
    #model = load_model('streetlanes1.hdf5')
    model = load_model('noisyLanesNet0.hdf5')
    modelSign = load_model('signNet0.hdf5')
    leftIndex = CNNFunc.findIndex(r'Data\lanesError\left')
    rightIndex = CNNFunc.findIndex(r'Data\lanesError\right')
    straightIndex = CNNFunc.findIndex(r'Data\lanesError\straight')
    
    # Connect to server
    ImageClient.connectClient('192.168.1.89', 50009)
    print('<INFO> Connection established, preparing to receive frames...')
    timeStart = time.time()
    ImageClient.sendCommand('SRT')
    
    # Receiving and processing frames
    while(1):
        # Start time of frame
        frameStart = time.time()
        
        # Receive and unload a frame
        imageData = ImageClient.receiveFrame()
        compressedImg = pickle.loads(imageData)
        image = cv2.imdecode(compressedImg, cv2.IMREAD_COLOR)
        
        # Preprocess the frame       
        procImg = preprocessors.removeTop(image, 3/5)      #cropout unnecessary parts    
        procImg = preprocessors.gray(procImg)                #change to gray scale  
        procImg = preprocessors.resizing(procImg, dimensions = (64, 64))
        procImg = preprocessors.extractWhite(procImg)           #extract whitelanes, result is binary image
        
        
        # Choose command based on result
        if np.mean(procImg)<5: #almost no street lines detected
            result ='STP'#stop
            #ImageClient.sendCommand('BYE')
            #break
        else:
            modelInput = np.expand_dims(procImg, axis = 0)
            modelInput = np.expand_dims(modelInput, axis = 3)
        
            # Predict result 
            prediction = model.predict(modelInput).argmax()
            if prediction == 0:
                result = 'LFT'#left
            elif prediction == 1:
                result = 'RGT'#right
            elif prediction == 2:
                result = 'STR'#straight
            else:
                print('<ERROR> Something went wrong, cannot determine whether result was left, right or straight')
                result = 'STP'#stop
        #ImageClient.sendCommand(result)
        
        # Initialize variables for sign detection
        normImg = CNNFunc.rangeChange(image, 255.0)
        windowGrid = np.empty((0, 20, 20, 3), 'uint8')
        windowLocs = np.empty((0, 2), int)
        winIndex = 0
        stopSigns = []
        forwardSigns = []
        
        # Create grid on image
        for (x, y, window) in CNNFunc.slideWindow(normImg, ROI=(0, 40, 320, 80), 
             windowSize=(20,20), stepSize=(20,20)):
            window = np.expand_dims(window, axis = 0)
            windowGrid = np.vstack((windowGrid, window))
            windowLocs = np.vstack((windowLocs, (x, y)))
        
        # Aplly signModel on grid and mark sign positions
        signPredictions = modelSign.predict(windowGrid)  
        for aPrediction in signPredictions:        
            aPrediction = aPrediction.argmax()
            if aPrediction == 1:
                (x, y) = windowLocs[winIndex]
                image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,0,255),1)
                stopSigns.append((x, y))
            if aPrediction == 2:
                (x, y) = windowLocs[winIndex]
                image = cv2.rectangle(image,(x,y),(x+19,y+19),(255,0,0),1)
                forwardSigns.append((x, y))
            winIndex += 1
        stopSigns = np.array(stopSigns)
        stopSigns = CNNFunc.removeSingleWin(stopSigns) # remove scatter false background detections
        print('stopSigns:', stopSigns)
        
        # Rough estimation of sign locations
        if len(stopSigns) > 0:            
            xS1 = np.min(stopSigns[:, 0])
            yS1 = np.min(stopSigns[:, 1])
            xS2 = np.max(stopSigns[:, 0])
            yS2 = np.max(stopSigns[:, 1])
            if xS1>10:
                xS1 = xS1 - 10
            if xS2<290:
                xS2 = xS2 + 29
            else:
                xS2 = xS2 + 19  
            yS1, yS2 = (yS1-10, yS2+29)  
            image = cv2.rectangle(image,(xS1,yS1), (xS2,yS2), (0,255,255), 1)
            
        forwardSigns = np.array(forwardSigns)
        forwardSigns = CNNFunc.removeSingleWin(forwardSigns) 
        print('forwardSigns:', forwardSigns)
        
        if len(forwardSigns) > 0:            
            x1 = np.min(forwardSigns[:, 0])
            y1 = np.min(forwardSigns[:, 1])
            x2 = np.max(forwardSigns[:, 0])
            y2 = np.max(forwardSigns[:, 1])
            if x1>10:
                x1 = x1 - 10
            if x2<290:
                x2 = x2 + 29
            else:
                x2 = x2 + 19  
            y1, y2 = (y1-10, y2+29)  
            image = cv2.rectangle(image,(x1,y1), (x2,y2), (255,0,255), 1)        
        
        # Show result on frame
        #cv2.putText(image, 'Direction: {}'.format(result), (10, 200), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        #cv2.imshow('Frame', image)
        
        #cv2.putText(procImg2, '{}-{:0.2f}'.format(result,prediction2), (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        #procImg = np.stack((procImg,)*3, -1)
        cv2.putText(procImg, '{}'.format(result), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)   
        cv2.imshow('Frame', procImg)
        key = cv2.waitKey(1) & 0xFF
        #cv2.waitKey(1)
        
        # Calculate framerate
        frameRate = 1 / (time.time() - frameStart)
        print('Frame rate is {0:.02f}fps'.format(frameRate))
        
        # Exit when q is pressed
        if key == ord('q'): 
            break
        else:
            if key == ord('a'):
                cv2.imwrite(r'Data\lanesError\left\left{:03}.jpg'.format(leftIndex), image)
            elif key == ord('s'):
                cv2.imwrite(r'Data\lanesError\straight\straight{:03}.jpg'.format(straightIndex), image)
            elif key == ord('d'):
                cv2.imwrite(r'Data\lanesError\right\right{:03}.jpg'.format(rightIndex), image)
            else:
                pass
            ImageClient.sendCommand(result)
            
        '''if key != 113: #ord(q)
            ImageClient.sendCommand(result)
        else:
            ImageClient.sendCommand('BYE')
            break'''
        
        #ImageClient.sendCommand(result)
    
    # Ending sequence
    #imageData = ImageClient.receiveOneImage()
    #image = pickle.loads(imageData)
    ImageClient.sendCommand('BYE')
    ImageClient.closeClient()
    
    elapsedTime = time.time() - timeStart
    print('<INFO> Total elapsed time is: ', elapsedTime)
    print('Press any key to exit the program')
    #cv2.imshow('Picture from server', image)
    cv2.waitKey(0)  
    
    
    
    
if __name__ == '__main__': main()
