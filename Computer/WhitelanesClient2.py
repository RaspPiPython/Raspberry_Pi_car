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
    
    #modelReg = load_model('regressWlanes2.hdf5')
    
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
        
        # Preprocess the frame       
        procImg = preprocessors.removeTop(image, 11/20)
        procImg = preprocessors.gray(procImg)
        #procImg2 = preprocessors.adjustDark(procImg)
        #procImg = preprocessors.extractWhite(procImg2)
        procImg2 = preprocessors.extractWhite(procImg)
        procImg = preprocessors.carBirdeyeView(procImg2, 5/12, 9/5, 160)
        procImg = preprocessors.resizing(procImg)
        #height, width = procImg2.shape[:2]
        #procImg = cv2.resize(procImg2,(int(32/160*width), int(32/160*height)), interpolation = cv2.INTER_CUBIC)
        #procImg = preprocessors.carBirdeyeView(procImg, 5/12, 9/5, 32)
        
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
        
        '''
        #------------------------------start regression
        croppedImg = preprocessors.removeTop(image, 13/20)      #cropout unnecessary parts    
        procImg3 = preprocessors.gray(croppedImg)                #change to gray scale
        procImg3 = preprocessors.extractWhite(procImg3)           #extract whitelanes, result is binary image
        procImg3 = preprocessors.resizing(procImg3)               #resize to 32x32 to feed into network
        
        procImg3 = np.expand_dims(procImg3, axis = 0)
        procImg3 = np.expand_dims(procImg3, axis = 3)
        
        prediction2 = modelReg.predict(procImg)
        print(prediction2, type(prediction2))
        prediction2 = float(prediction2)
        #------------------------------end regression
        '''
        
        # Show result on frame
        #cv2.putText(image, 'Direction: {}'.format(result), (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        #cv2.imshow('Frame', image)
        #cv2.putText(procImg2, '{}-{:0.2f}'.format(result,prediction2), (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(procImg2, '{}'.format(result), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)   
        cv2.imshow('Frame', procImg2)
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