# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:20:47 2018

@author: tranl
"""

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
import tkinter as tk

def main():
    global master
    master = tk.Tk()
    
    global connectButton
    connectButton = tk.Button(master, text='Connect', command=connectionInit)
    connectButton.pack()
    
    global autoButton
    autoButton = tk.Button(master, text='Auto mode', command=controlFunc)
    autoButton.pack()
    
    global manualButton
    manualButton = tk.Button(master, text='Manual mode', command=manualFunc)
    manualButton.pack()
    
    exitButton = tk.Button(master, text='Exit', command=master.destroy)
    exitButton.pack()    
    
    global textBox
    textBox = tk.Text(master, height=2, width=40)
    textScroll = tk.Scrollbar(master)
    textBox.pack(side=tk.LEFT, fill=tk.Y)
    textScroll.pack(side=tk.RIGHT, fill=tk.Y)
    textBox.config(yscrollcommand=textScroll.set)
    textScroll.config(command=textBox.yview)    
    textBox.insert(tk.END, 'Press Connect to establish connection.\nRemember to check if the server is on.')
    
    
    # tk GUI main loop ends here
    status = connectButton.cget('text')
    master.mainloop()    
    
    # This part run after the main loop ended
    
    if status == 'Connect': #check if user forgot to disconnect to server
        ImageClient.sendCommand('BYE')
        ImageClient.closeClient()
    else:
        print('aaaaaa')
    print('Main window closed')
    
def disconnecting():
    print('<INFO> Disconnecting...')
    ImageClient.sendCommand('BYE')
    ImageClient.closeClient()
    print('<INFO> Disconnected.')
    textBox.delete('1.0', tk.END)
    textBox.config(height=2)
    textBox.insert(tk.END, 'Press Connect to establish connection.\nRemember to check if the server is on.')
    connectButton.config(text='Connect')
    connectButton.config(command=connectionInit)
    
    
def connectionInit():    
    # Intialize connection
    global ImageClient
    ImageClient = PiImageClient()
    ImageClient.connectClient('192.168.1.89', 50009)
    print('<INFO> Connection established, preparing to receive frames...')
        
    # Initialize preprocessor functions and CNN models
    print('<INFO> Initializing models...')
    global preprocessors
    preprocessors = CNNFunc.preprocessors
    global modelLanes
    modelLanes = load_model('noisyLanesNet0.hdf5')
    global modelSign
    modelSign = load_model('signNet1.hdf5')
    # Change text in textBox field
    textBox.delete('1.0', tk.END)
    connectButton.config(text='Disconnect')
    textBox.config(height=6)
        
    userGuide = '''Connection established. 
Please choose a mode.
Click the image to enable control.      
In auto mode:
Q: Pause and return to mode selection
R: Toggle runFlag RN
T: Toggle whiteLanesFlag WL
Z: Toggle trafficSignFlag TS
X: Save current frame, unprocessed
In manual mode:
Q: Pause and return to mode selection
W: Forward
S: Backward
A: Turn left
D: Turn right
'''
    connectButton.config(command=disconnecting)
    textBox.insert(tk.END, userGuide)
        
    print('<INFO> Initialization finished. Please choose a mode.')
    
    

# Function for manual mode    
def manualFunc():
    # Change button text
    print('Manual function started.')
    #textBox.delete('1.0', tk.END)
    #textBox.insert(tk.END, 'Manual mode started.')
    prevFrame = ImageClient.counter
    timeStart = time.time()
    frameStart = 0.0
    command = 'STP'
    #oldCommand = 'STP'
    ImageClient.sendCommand('MAN')
    print('<INFO> Running...')
    
    while(1):
        # Calculate framerate
        frameRate = 1 / (time.time() - frameStart)
        print('Frame rate is {0:.02f}fps'.format(frameRate))
        frameStart = time.time()
        
        # Receive and unload a frame
        imageData = ImageClient.receiveFrame()
        compressedImg = pickle.loads(imageData)
        image = cv2.imdecode(compressedImg, cv2.IMREAD_COLOR)
        
        cv2.imshow('Frame', image)
        key = cv2.waitKey(1) & 0xFF
        #cv2.waitKey(1)
        #cv2.putText()
        
        # Choose command based on the key pressed
        if key == 255:
            pass
        elif key == ord('q'): 
            break
        elif key == ord('w'):
            command = 'STR'
        elif key == ord('a'):
            command = 'LFT'
        elif key == ord('s'):
            command = 'BCK'
        elif key == ord('d'):
            command = 'RGT'
        elif key == ord('b'):
            command = 'STP'
        else:
            pass
        ImageClient.sendCommand(command)
    
    # Print some statistics
    ImageClient.sendCommand('PAU')    
    #ImageClient.counter = 0 # reset image counter
    print('<INFO> Number of frames during this run: {}'.format(ImageClient.counter-prevFrame))
    elapsedTime = time.time() - timeStart
    print('<INFO> Total elapsed time is: ', elapsedTime)
    print('<INFO> Manual mode suspended, you can now choose another mode.')
    
    

# Function for automatic mode
def controlFunc():
    print('<INFO> Auto function started.')
    
    # Initialization
    print('<INFO> Initializing...')
    #global ImageClient
    #ImageClient = PiImageClient()
    #preprocessors = CNNFunc.preprocessors
    #model = load_model('streetlanes1.hdf5')
    #modelLanes = load_model('noisyLanesNet0.hdf5')
    #modelSign = load_model('signNet0.hdf5')
    
    # Uncoment these lines to initialize variables used for gathering misclassified images
    #leftIndex = CNNFunc.findIndex(r'Data\lanesError\left')
    #rightIndex = CNNFunc.findIndex(r'Data\lanesError\right')
    #straightIndex = CNNFunc.findIndex(r'Data\lanesError\straight')
    signIndex = CNNFunc.findIndex(r'Data\signError')
    prevFrame = ImageClient.counter
    whiteThreshold = 210
    meanThreshold = 35
    meanProc = 0.0
    frameStart = 0.0
    runFlag = False
    whiteLanesFlag = False
    trafficSignFlag = False
    
    
    # Connect to server
    #ImageClient.connectClient('192.168.1.89', 50009)
    #print('<INFO> Connection established, preparing to receive frames...')
    timeStart = time.time()
    ImageClient.sendCommand('SRT')
    print('<INFO> Running...')
    
    # Receiving and processing frames
    while(1):
        # Calculate framerate
        frameRate = 1 / (time.time() - frameStart)
        print('Frame rate is {0:.02f}fps'.format(frameRate))
        frameStart = time.time()
        
        # Receive and unload a frame
        imageData = ImageClient.receiveFrame()
        compressedImg = pickle.loads(imageData)
        image = cv2.imdecode(compressedImg, cv2.IMREAD_COLOR)
        
        # Preprocess the frame       
        procImg0 = preprocessors.removeTop(image, 3/5)      #cropout unnecessary parts    
        procImg0 = preprocessors.gray(procImg0)                #change to gray scale  
        procImg0 = preprocessors.resizing(procImg0, dimensions = (64, 64))
        procImg = preprocessors.extractWhite(procImg0, threshold = (whiteThreshold, 255)) #extract whitelanes, result is binary image
        meanProc = np.mean(procImg)
        
        if meanProc > meanThreshold: # possible noises from light -> increase threshold to counter
            whiteThreshold = 230
            procImg = preprocessors.extractWhite(procImg0, threshold = (whiteThreshold, 255))
        
        # Detect whitelanes section
        if meanProc < 5: #stop when almost no lanes detected
            result ='STP'
            #ImageClient.sendCommand('BYE')
            #break
        else:
            if whiteLanesFlag == True:
                modelInput = np.expand_dims(procImg, axis = 0)
                modelInput = np.expand_dims(modelInput, axis = 3)
        
                # Predict result 
                prediction = modelLanes.predict(modelInput).argmax()
                if prediction == 0:
                    result = 'LFT'#left
                elif prediction == 1:
                    result = 'RGT'#right
                elif prediction == 2:
                    result = 'STR'#straight
                else:
                    print('<ERROR> Something went wrong, cannot determine whether result was left, right or straight')
                    result = 'STP'#stop
            else:
                result = 'STP'
        
        
        # Detect traffic sign section
        if trafficSignFlag == True:
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
            #print('stopSigns:', stopSigns)
            
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
            #print('forwardSigns:', forwardSigns)
            
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
        procImg = np.vstack((np.zeros((176, 64)).astype('uint8'), procImg))
        stackedProcImg = np.stack((procImg,)*3, -1)
        procImg2 = np.hstack((image, stackedProcImg))
        
        
        # Show result and flag values
        cv2.putText(procImg2, '{}'.format(result), (325, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  
        cv2.putText(procImg2, 'RN: {}'.format(int(runFlag)), (325, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
        cv2.putText(procImg2, 'WL: {}'.format(int(whiteLanesFlag)), (325, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
        cv2.putText(procImg2, 'TS: {}'.format(int(trafficSignFlag)), (325, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
        cv2.putText(procImg2, '{0:.02f}'.format(frameRate), (325, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
        cv2.imshow('Frame', procImg2)
        key = cv2.waitKey(1) & 0xFF
        #cv2.waitKey(1)
        #cv2.putText()
        
        # Choose command based on the key pressed
        if key == ord('q'): 
            break
        # Uncomment the lines below to gather data when image is misclassified
        #elif key == ord('a'):
        #    cv2.imwrite(r'Data\lanesError\left\left{:03}.jpg'.format(leftIndex), image)
        #elif key == ord('s'):
        #    cv2.imwrite(r'Data\lanesError\straight\straight{:03}.jpg'.format(straightIndex), image)
        #elif key == ord('d'):
        #    cv2.imwrite(r'Data\lanesError\right\right{:03}.jpg'.format(rightIndex), image)
        elif key == ord('r'):
            runFlag = not runFlag
            result = 'RFL'
        elif key == ord('t'):
            whiteLanesFlag = not whiteLanesFlag
        elif key == ord('z'):
            trafficSignFlag = not trafficSignFlag
        elif key == ord('x'):
            imageSave = cv2.imdecode(compressedImg, cv2.IMREAD_COLOR)
            cv2.imwrite(r'Data\signError\signErr{:03}.jpg'.format(signIndex), imageSave)
            signIndex += 1
            cv2.imwrite(r'Data\signError\signErrPosition{:03}.jpg'.format(signIndex), procImg2)
            signIndex += 1
        else:
            pass
            #print('Invalid command {}, please press another button'.format(key))
        
        # Send the command
        ImageClient.sendCommand(result)
            
        '''if key != 113: #ord(q)
            ImageClient.sendCommand(result)
        else:
            ImageClient.sendCommand('BYE')
            break'''
        
        #ImageClient.sendCommand(result)
    
    # Suspending connection
    #imageData = ImageClient.receiveOneImage()
    #image = pickle.loads(imageData)
    ImageClient.sendCommand('PAU')    
    #ImageClient.counter = 0 # reset image counter
    print('<INFO> Number of frames during this run: {}'.format(ImageClient.counter-prevFrame))
    print('<INFO> Auto mode suspended, you can now choose another mode.')
    #cv2.imshow('Picture from server', image)
    #cv2.waitKey(0)  
    
    # Print some statistics
    elapsedTime = time.time() - timeStart
    print('<INFO> Total elapsed time is: ', elapsedTime)
    
    
if __name__ == '__main__': main()
