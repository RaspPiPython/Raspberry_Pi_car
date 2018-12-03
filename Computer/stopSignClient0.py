# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:15:58 2018

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
    # Initialize
    ImageClient = PiImageClient()
    #preprocessors = CNNFunc.preprocessors
    #model = load_model('stopSign3.hdf5')
    model = load_model('signNet0.hdf5')
    
    # Connect to server
    ImageClient.connectClient('192.168.1.89', 50009)
    print('<INFO> Connection established, preparing to receive frames...')
    timeStart = time.time()
    ImageClient.sendCommand('SRT')
    
    # Receiving and processing frames
    while(1):
        # Receive and unload a frame
        imageData = ImageClient.receiveFrame()
        compressedImg = pickle.loads(imageData)
        image = cv2.imdecode(compressedImg, cv2.IMREAD_COLOR)
        procImg = CNNFunc.rangeChange(image, 255.0)
        windowGrid = np.empty((0, 20, 20, 3), 'uint8')
        windowLocs = np.empty((0, 2), int)
        winIndex = 0
        
        for (x, y, window) in CNNFunc.slideWindow(procImg, ROI=(0, 20, 320, 80), 
             windowSize=(20,20), stepSize=(20,20)):
            window = np.expand_dims(window, axis = 0)
            windowGrid = np.vstack((windowGrid, window))
            windowLocs = np.vstack((windowLocs, (x, y)))
            
        predictions = model.predict(windowGrid)  
        for aPrediction in predictions:        
            aPrediction = aPrediction.argmax()
            if aPrediction != 0 and aPrediction > 0.9:
                (x, y) = windowLocs[winIndex]
                if aPrediction == 1:                    
                    image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,0,255),1)
                elif aPrediction == 2:
                    image = cv2.rectangle(image,(x,y),(x+19,y+19),(255,0,0),1)
            winIndex += 1
        
        result = 'STP'
        
        #cv2.putText(image, '{}'.format(result), (10, 30), 
         #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)   
        cv2.imshow('Frame', image)
        key = cv2.waitKey(1) & 0xFF
        #cv2.waitKey(1)
        
        # Exit when q is pressed
        if key == ord('q'): 
            break
        else:
            ImageClient.sendCommand(result)
            
        '''if key != 113: #ord(q)
            ImageClient.sendCommand(result)
        else:
            ImageClient.sendCommand('BYE')
            break'''
        
        #ImageClient.sendCommand(result)
    
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
