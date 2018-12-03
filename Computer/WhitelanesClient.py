# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:02:30 2018

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
    preprocessors = CNNFunc.preprocessors
    model = load_model('streetlanes1.hdf5')
    
    # Connect to server
    ImageClient.connectClient('192.168.1.89', 50009)
    print('<INFO> Connection established, preparing to receive frames...')
    timeStart = time.time()
    
    # Receiving and processing frames
    while(1):
        # Receive and unload a frame
        imageData = ImageClient.receiveFrame()
        image = pickle.loads(imageData)
        
        # Preprocess the frame       
        procImg = preprocessors.removeTop(image, 11/20)
        procImg = preprocessors.gray(procImg)
        #procImg2 = preprocessors.adjustDark(procImg)
        #procImg = preprocessors.extractWhite(procImg2)
        procImg2 = preprocessors.extractWhite(procImg)
        procImg = preprocessors.carBirdeyeView(procImg2, 5/12, 9/5, 160)
        procImg = preprocessors.resizing(procImg)
        
        # Choose command based on result
        if np.mean(procImg)<2: #almost no street lines detected
            result ='STP'#stop
            #ImageClient.sendCommand('BYE')
            #break
        else:
            procImg = np.expand_dims(procImg, axis = 0)
            procImg = np.expand_dims(procImg, axis = 3)
        
            # Predict result 
            prediction = model.predict(procImg).argmax()
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
        
        # Show result on frame
        #cv2.putText(image, 'Direction: {}'.format(result), (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        cv2.putText(procImg2, 'Direction: {}'.format(result), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)            
        cv2.imshow('Frame', procImg2)
        #cv2.imshow('Frame', procImg2)
        key = cv2.waitKey(1) & 0xFF
        #cv2.waitKey(1)
        
        # Exit when q is pressed
        if key == ord('q'):            
            break
        #else:
        #    ImageClient.sendCommand(result)
            
        '''if key != 113: #ord(q)
            ImageClient.sendCommand(result)
        else:
            ImageClient.sendCommand('BYE')
            break'''
        
        ImageClient.sendCommand(result)
    
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