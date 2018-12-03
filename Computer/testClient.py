# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:30:43 2018

@author: tranl
"""

from SupFunctions.ServerClientFunc import PiImageClient
#from SupFunctions import CNNFunc
import time
import pickle
import cv2
import socket
import struct
import io
from PIL import Image
#import numpy as np

def main():
    # Initialize
    result = 'STP'
    ImageClient = PiImageClient()
    #preprocessors = CNNFunc.preprocessors
    #model = load_model('streetlanes1.hdf5')
    
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
        #if result == 'STP':
        #    result = 'STR'
        #else:
        #    result = 'STP'
        # Preprocess the frame  
        ImageClient.sendCommand(result)
        
        cv2.imshow('Frame', image)
        key = cv2.waitKey(1) & 0xFF
        
        # Exit when q is pressed
        if key == ord('q'):
            #ImageClient.sendCommand('BYE')
            '''response = ImageClient.receiveCommand()
            if response == 'ACK':
                print('BYE successfully received')
            else:
                print('<WARNING> BYE is not received')'''
            break
        #else:
        #    ImageClient.sendCommand(result)
    
    ImageClient.closeClient()    
    elapsedTime = time.time() - timeStart 
    print('<INFO> Total elapsed time is: ', elapsedTime)
    print('Press any key to exit the program')
    cv2.waitKey(0)  
    
    
def main3():
    # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
    # all interfaces)
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 8000))
    server_socket.listen(0)

    # Accept a single connection and make a file-like object out of it
    connection = server_socket.accept()[0].makefile('rb')
    try:
        while True:
            # Read the length of the image as a 32-bit unsigned int. If the
            # length is zero, quit the loop
           image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
           if not image_len:
               break
           # Construct a stream to hold the image data and read the image
           # data from the connection
           image_stream = io.BytesIO()
           image_stream.write(connection.read(image_len))
           # Rewind the stream, open it as an image with PIL and do some
           # processing on it
           image_stream.seek(0)
           image = Image.open(image_stream)
           print('Image is %dx%d' % image.size)
           image.verify()
           print('Image is verified')
    finally:
        connection.close()
        server_socket.close()

def main2():
    # Initialize
    result = 'STP'
    ImageClient = PiImageClient()
    ImageClient2 = PiImageClient()
    #preprocessors = CNNFunc.preprocessors
    #model = load_model('streetlanes1.hdf5')
    
    # Connect to server
    ImageClient.connectClient('192.168.1.89', 50009)
    ImageClient2.connectClientNODELAY('192.168.1.89', 50002)
    print('<INFO> Connection established, preparing to receive frames...')
    timeStart = time.time()
    
    ImageClient2.sendCommand('STR')
    
    # Receiving and processing frames
    while(1):
        # Receive and unload a frame
        imageData = ImageClient.receiveFrame()
        image = pickle.loads(imageData)
        if result == 'STP':
            result = 'STR'
        else:
            result = 'STP'
        # Preprocess the frame       
        '''procImg = preprocessors.removeTop(image, 11/20)
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
        #cv2.imshow('Frame', procImg2)'''
        cv2.imshow('Frame', image)
        key = cv2.waitKey(25) & 0xFF
        #ImageClient2.sendCommand(result)
        #ImageClient2.sendCommand(result)
        
        # Exit when q is pressed
        if key == ord('q'):
            ImageClient2.sendCommand('BYE')
            '''response = ImageClient.receiveCommand()
            if response == 'ACK':
                print('BYE successfully received')
            else:
                print('<WARNING> BYE is not received')'''
            break
        else:
            ImageClient2.sendCommand(result)
            
        #ImageClient2.sendCommand(result)
    
    #imageData = ImageClient.receiveOneImage()
    #image = pickle.loads(imageData)
    ImageClient.closeClient()
    ImageClient2.closeClient()
    
    elapsedTime = time.time() - timeStart 
    print('<INFO> Total elapsed time is: ', elapsedTime)
    print('Press any key to exit the program')
    #cv2.imshow('Picture from server', image)
    cv2.waitKey(0)  
    
    
    
    
if __name__ == '__main__': main()