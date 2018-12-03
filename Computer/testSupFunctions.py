# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:55:33 2018

@author: tranl
"""

from SupFunctions import CNNFunc
import cv2
#import numpy as np

def main():
    preprocessors = CNNFunc.preprocessors
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\left\013.jpg')
    
    # Preprocess the frame       
    procImg = preprocessors.removeTop(image, 11/20)
    procImg = preprocessors.gray(procImg)
    #procImg2 = preprocessors.adjustDark(procImg)
    #procImg = preprocessors.extractWhite(procImg2)
    procImg = preprocessors.extractWhite(procImg)
    procImg = preprocessors.carBirdeyeView(procImg, 5/12, 9/5, 160)
    procImg = preprocessors.resizing(procImg)
    
    

    # Preprocess the frame       
    procImg2 = preprocessors.removeTop(image, 11/20)
    procImg2 = preprocessors.gray(procImg2)
    #procImg2 = preprocessors.adjustDark(procImg)
    #procImg = preprocessors.extractWhite(procImg2)
    procImg2a = preprocessors.extractWhite(procImg2)
    height, width = procImg2a.shape[:2]
    #procImg2b = cv2.resize(procImg2a,(int(32/160*width), int(32/160*height)), interpolation = cv2.INTER_AREA)
    procImg2b = cv2.resize(procImg2a,(int(32/160*width), int(32/160*height)), interpolation = cv2.INTER_CUBIC)
    procImg2c = preprocessors.extractWhite(procImg2b, threshold = (50, 255))
    procImg2d = preprocessors.carBirdeyeView2(procImg2c, 5/12, 9/5, 32)
    
    
    cv2.imshow('no resize before birdeye', procImg)
    cv2.waitKey(0)
    cv2.imshow('resize before birdeye a', procImg2a)
    cv2.waitKey(0)
    cv2.imshow('resize before birdeye b', procImg2b)
    cv2.waitKey(0)
    cv2.imshow('resize before birdeye c', procImg2c)
    cv2.waitKey(0)
    cv2.imshow('resize before birdeye d', procImg2d)
    cv2.waitKey(0)


def main3():
    image = cv2.imread(r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\TrafficSigns\lightsSign\001.jpg')
    i = 0
    for (x, y, window) in CNNFunc.slideWindow(image):
        i += 1
        image = cv2.rectangle(image,(x,y),(x+19,y+19),(0,255,0),1)
        #cv2.imshow('Frame {}'.format(i), window)
        #cv2.waitKey(0)
    cv2.imshow('Img', image)
    cv2.waitKey(0)

def main2():
    preprocessors = CNNFunc.preprocessors
    img = cv2.imread(r'C:\Users\tranl\JupyterNotebooks\FirstBook\index.jpg')
    gray = preprocessors.gray(img)
    brighten = preprocessors.adjustDark(gray, 1.2)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.imshow('gray', gray)
    cv2.waitKey()
    cv2.imshow('bright', brighten)
    cv2.waitKey()
    
if __name__ == '__main__': main()