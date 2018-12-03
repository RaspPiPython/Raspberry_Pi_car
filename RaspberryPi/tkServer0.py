import socket
import pickle
import time
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from SendFrameInOO import PiImageServer
import CarControlFunc 

def main():
    # initialize the server and time stamp
    ImageServer = PiImageServer()    
    ImageServer.openServer('192.168.1.89', 50009)

    # Initialize the camera object
    camera = PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 15 
    camera.exposure_mode = 'sports' #reduce blur
    rawCapture = PiRGBArray(camera)
    jpgQuality = 90
    #pngCompression = 3
    
    # allow the camera to warmup
    time.sleep(1)

    # initialize GPIO pins and variables for controlling car
    runFlag = False
    Control = CarControlFunc.PiCarCtrl()
    oldCommand = 'STR'
    angleIncrement = 0
    defaultLeft = 6.2
    defaultRight = 7.6

    # capture frames from the camera
    print('<INFO> Preparing to stream video...')
    timeStart = time.time()

    
    for frame in camera.capture_continuous(rawCapture, format="bgr",
                                           use_video_port = True):
        # receive command from laptop and print it
        command = ImageServer.recvCommand()

        # control the car based on the command
        # if runFlag is not True, car will not move
        if runFlag == True:            
            if command == 'STR':
                if oldCommand == 'STR':
                    Control.forward(speed = 44)
                else:
                    Control.forward(speed = 45)
            elif command == 'LFT':
                #if oldCommand == 'STR':
                #    Control.brake(brakeTime=0.1)
                if oldCommand == 'LFT':
                    if angleIncrement < 0.6:
                        angleIncrement += 0.12                
                else:
                    angleIncrement = 0
                angle = defaultLeft - angleIncrement
                Control.left(angle)
            elif command == 'RGT':
                #if oldCommand == 'STR':
                #    Control.brake(brakeTime=0.1)
                if oldCommand == 'RGT':
                    if angleIncrement < 0.6:
                        angleIncrement += 0.2
                else:
                    angleIncrement = 0
                angle = defaultRight + angleIncrement
                Control.right(angle)
            elif command == 'STP':
                if oldCommand != 'STP':
                    Control.brake()
                else:
                    Control.idle()
            elif command == 'BCK':
                Control.backward(speed = 45)                
            oldCommand = command
            
            if command == 'RFL':
                Control.brake()

        # if PAU is received, pausing until SRT or BYE is received    
        if command == 'PAU':
            print('Pausing, waiting for SRT or BYE command...')
            runFlag = False
            Control.brake()
            while(1):
                command = ImageServer.recvCommand()
                if command == 'SRT' or command == 'MAN' or command == 'BYE':
                    break
                else:
                    time.sleep(1)

        # other commands
        if command == 'BYE':
            print('BYE received, ending stream session...')
            break
        elif command == 'SRT':
            print('SRT received, start streaming...')
        elif command == 'MAN':
            runFlag = True
            print('MAN received, start manual mode...')
        elif command == 'RFL':
            runFlag = not runFlag
            
        
        
        # grab the raw NumPy array representing the image, then initialize 
        # the timestamp and occupied/unoccupied text
        image = frame.array               
        #cv2.imshow('Frame', image)
        #key = cv2.waitKey(1) & 0xFF # catch any key input
        #print(image[0])

        # compress the raw image to jpg with specified quality
        ret, compressedImg = cv2.imencode('.jpg', image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpgQuality])
        #ret, compressedImg = cv2.imencode('.png', image,
        #                [int(cv2.IMWRITE_PNG_COMPRESSION), pngCompression])
        imageData = pickle.dumps(compressedImg) 
        ImageServer.sendFrame(imageData) # send the frame data            

        # clear the stream in preparation for the next one
        rawCapture.truncate(0)       

        # if the 'q' key is pressed, break from the loop
        #if key == ord("q"):           
        #    break
    print('<INFO> Video stream ended')
    ImageServer.closeServer()

    elapsedTime = time.time() - timeStart
    print('<INFO> Total elapsed time is: ', elapsedTime)

    


if __name__ == '__main__': main()
