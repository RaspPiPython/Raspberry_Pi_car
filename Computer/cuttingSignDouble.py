# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 23:27:22 2018

@author: tranl
"""



import numpy as np
import cv2
import random
from SupFunctions import CNNFunc


'''
Cropping 320x240 sections that contains stop sign to create data for localization network
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main4():
    locRegionList = np.array([  # location list of the signs in the images
            (1, 33, 44, 58, 56), (2, 62, 54, 51, 50), (3, 95, 64, 44, 44), (4, 120, 72, 39, 39), (5, 146, 80, 34, 34), 
            (6, 159, 80, 34, 34), (7, 167, 77, 36, 36), (8, 180, 74, 38, 38), (9, 195, 71, 40, 40), (10, 212, 68, 44, 42),
            (11, 227, 63, 46, 46), (12, 232, 56, 52, 50), (13, 248, 45, 58, 57), (14, 20, 36, 62, 62), (15, 127, 45, 56, 57),            
            (16, 66, 38, 51, 63), (17, 127, 49, 47, 55), (18, 177, 47, 36, 57), (19, 199, 65, 37, 44), (20, 220, 76, 26, 37),
            (21, 87, 73, 33, 39), (22, 119, 72, 36, 39), (23, 162, 73, 33, 39), (24, 184, 68, 32, 42), (25, 33, 67, 44, 43),
            (26, 148, 65, 41, 44), (27, 76, 58, 44, 48), (28, 57, 54, 41, 50), (29, 31, 47, 39, 55), (30, 173, 68, 30, 42),
            (31, 192, 64, 39, 45), (32, 215, 60, 48, 48), (33, 240, 57, 51, 50), (34, 249, 54, 55, 51), (35, 226, 67, 36, 43),
            (36, 96, 62, 46, 45), (37, 64, 54, 52, 50), (38, 122, 42, 59, 59), (39, 38, 57, 48, 48), (40, 38, 64, 45, 44),
            (41, 65, 49, 54, 54), (42, 89, 53, 52, 51), (43, 79, 79, 34, 34), (44, 104, 80, 35, 34), (45, 73, 83, 32, 32),
            (46, 59, 69, 42, 41), (47, 78, 54, 52, 51), (48, 106, 79, 34, 34), (49, 86, 88, 28, 29), (50, 28, 74, 39, 37),
            (51, 43, 63, 46, 45), (52, 119, 75, 38, 37), (53, 128, 83, 31, 32), (54, 82, 83, 31, 32), (55, 71, 54, 53, 51),
            (56, 155, 57, 49, 49), (57, 183, 70, 41, 40), (58, 197, 66, 44, 43), (59, 207, 76, 37, 37), (60, 168, 81, 34, 34),
            (61, 232, 68, 44, 42), (62, 153, 62, 46, 46), (63, 200, 51, 55, 53), (64, 191, 67, 44, 43), (65, 182, 87, 30, 29),
            (66, 197, 70, 41, 40), (67, 238, 76, 38, 36), (68, 231, 65, 46, 44), (69, 172, 68, 42, 42), (70, 220, 61, 47, 47),
            (71, 189, 62, 45, 46), (72, 243, 44, 60, 57), (73, 227, 62, 48, 46), (74, 200, 71, 41, 40), (75, 169, 53, 52, 52)
            ])
    locRegionListTest = np.array([
            (1, 33, 44, 58, 56), (2, 62, 54, 51, 50)
            ])
    counter = 0
    counterTotal = 0
    #label = imLink.split(os.path.sep)[-1][0:3]
    for locReg in locRegionList:
        (imName, x0, y0, imWidth, imHeight) = locReg
        imLink = r'Data\localizationRaw\0stop\{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(0, 0, 340, 255), windowSize=(320,240), stepSize=(5,5)):
            counter += 1
            # (x1, y1) is the top left corner of the sign in each 320x240 image cropped from the 340x255 one
            x1 = x0 - x 
            y1 = y0 - y
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.9, 1.1)).astype('int')
            window[window>255] = 255
            window = window.astype('uint8')
                
            cv2.imwrite(r'Data\localization\0stopSign\{:03}{:03}{:03}{:03}.png'.format(x1, y1, imWidth, imHeight), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of stop sign sections created is {}'.format(len(locRegionList), counterTotal))
    

'''
Cropping 19x19 (then resize to 20x20) or 20x20 sections that contains part of windows that are recognized incorrectly
For these images: max distance from edge of cropping ROI to edge of sign is 4-5 pixels
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main3():
    croppingRegionList = np.array([(1, 137, 37, 26, 26), (1, 277, 77, 26, 26)])
    counter = 0
    counterTotal = 0
    #label = imLink.split(os.path.sep)[-1][0:3]
    for cropReg in croppingRegionList:
        (imName, x0, y0, imWidth, imHeight) = cropReg
        imLink = r'Data\stopSignDouble\Errors\err{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(x0, y0, imWidth, imHeight), windowSize=(20,20), stepSize=(2,2)):
            counter += 1
            
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.95, 1.05)).astype('int')
            winArray = window.flatten()
            for i in range(0, len(winArray)):
                if winArray[i] > 255:
                    winArray[i] = 255
            winArray = winArray.astype('uint8')
            window = np.reshape(winArray, (20, 20, 3))
                
            cv2.imwrite(r'Data\stopSignDouble\croppedErrorSections\err{:03}{:03}{:03}.png'.format(imName, x, y), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of stop sign sections created is {}'.format(len(croppingRegionList), counterTotal))
    

'''
Cropping 19x19 (then resize to 20x20) or 20x20 sections that contains part of forward sign
For these images: max distance from edge of cropping ROI to edge of sign is 4-5 pixels
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main2():
    fullRegionList = np.array([(1, 49, 20, 76, 76), (2, 66, 29, 70, 70), (3, 71, 39, 62, 62), (4, 78, 44, 58, 58),
                                   (5, 99, 54, 52, 52), (6, 77, 55, 50, 50), (7, 89, 60, 46, 46), (8, 82, 63, 44, 44),
                                   (9, 92, 64, 44, 44), (10, 124, 67, 42, 42), (11, 4, 42, 60, 60), (12, 38, 50, 54, 54),
                                   (13, 26, 55, 50, 50), (14, 47, 59, 48, 48), (15, 48, 63, 44, 44), (16, 198, 18, 78, 78),
                                   (17, 160, 28, 72, 72), (18, 179, 37, 64, 64), (19, 169, 44, 58, 58), (20, 191, 51, 54, 54),
                                   (21, 214, 18, 76, 76), (22, 248, 47, 56, 56), (23, 211, 58, 48, 48), (24, 131, 59, 48, 48),
                                   (25, 206, 67, 42, 42), (26, 249, 56, 50, 50), (27, 225, 63, 44, 44), (28, 252, 68, 40, 40),
                                   (29, 232, 69, 40, 40), (30, 233, 71, 38, 38), (31, 133, 61, 36, 46), (32, 49, 51, 52, 52),
                                   (33, 3, 38, 64, 62), (34, 201, 46, 36, 56), (35, 276, 38, 44, 62), (36, 110, 44, 48, 58),
                                   (37, 25, 37, 50, 62), (38, 1, 33, 52, 66), (39, 193, 46, 54, 56), (40, 252, 33, 68, 66)
                                    ])
    croppingRegionList = np.array([ (3, 71, 39, 62, 62), (4, 78, 44, 58, 58),
                                   (5, 99, 54, 52, 52), (6, 77, 55, 50, 50), (7, 89, 60, 46, 46), (8, 82, 63, 44, 44),
                                   (9, 92, 64, 44, 44), (10, 124, 67, 42, 42), (12, 38, 50, 54, 54),
                                   (13, 26, 55, 50, 50), (14, 47, 59, 48, 48), (15, 48, 63, 44, 44), 
                                   (18, 179, 37, 64, 64), (20, 191, 51, 54, 54),
                                   (22, 248, 47, 56, 56), (23, 211, 58, 48, 48), (24, 131, 59, 48, 48),
                                   (25, 206, 67, 42, 42), (26, 249, 56, 50, 50), (27, 225, 63, 44, 44), (28, 252, 68, 40, 40),
                                   (29, 232, 69, 40, 40), (30, 233, 71, 38, 38), (31, 133, 61, 36, 46), (32, 49, 51, 52, 52),
                                   (33, 3, 38, 64, 62), (34, 201, 46, 36, 56), (35, 276, 38, 44, 62), (36, 110, 44, 48, 58),
                                   (37, 25, 37, 50, 62), (38, 1, 33, 52, 66), (39, 193, 46, 54, 56)
                                    ])
    #imLink = r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\stopSignDouble\007.jpg' 
    #image = cv2.imread(imLink)
    counter = 0
    counterTotal = 0
    #label = imLink.split(os.path.sep)[-1][0:3]
    for cropReg in croppingRegionList:
        (imName, x0, y0, imWidth, imHeight) = cropReg
        imLink = r'Data\forwardSignDouble\{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(x0, y0, imWidth, imHeight), windowSize=(20,20), stepSize=(2,2)):
            counter += 1
            
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.95, 1.05)).astype('int')
            winArray = window.flatten()
            for i in range(0, len(winArray)):
                if winArray[i] > 255:
                    winArray[i] = 255
            winArray = winArray.astype('uint8')
            window = np.reshape(winArray, (20, 20, 3))
                
            cv2.imwrite(r'Data\forwardSignDouble\croppedForwardSignSections\{:03}{:03}{:03}.png'.format(imName, x, y), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of forward sign sections created is {}'.format(len(croppingRegionList), counterTotal))
    
    
'''
Cropping 19x19 (then resize to 20x20) or 20x20 sections that contains part of stop sign
For these images: max distance from edge of cropping ROI to edge of sign is 4-5 pixels
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main1():
    croppingRegionList = np.array([(1, 22, 64, 50, 50), (2, 35, 70, 46, 46), (3, 49, 77, 42, 42), (4, 55, 79, 40, 40),
                          (5, 60, 82, 38, 38), (6, 64, 85, 36, 36), (7, 68, 87, 34, 34), (8, 71, 88, 34, 34),
                          (9, 82, 91, 32, 32), (10, 253, 64, 52, 52), (11, 248, 67, 50, 50), (12, 244, 68, 50, 50),
                          (13, 242, 71, 48, 48), (14, 237, 73, 46, 46), (15, 232, 77, 44, 44), (16, 226, 78, 44, 44),
                          (17, 224, 82, 40, 40), (18, 220, 83, 40, 40), (19, 216, 86, 38, 38), (20, 217, 89, 36, 36),
                          (21, 116, 49, 60, 60), (22, 118, 60, 52, 52), (23, 120, 66, 48, 48), (24, 122, 70, 44, 44),
                          (25, 123, 75, 40, 40), (26, 124, 79, 38, 38), (27, 126, 84, 36, 36), (28, 164, 68, 46, 46),
                          (29, 88, 43, 62, 62), (30, 48, 52, 56, 56), (31, 161, 74, 40, 40), (32, 227, 50, 46, 58),
                          (33, 265, 50, 46, 58), (34, 102, 64, 46, 48), (35, 139, 56, 52, 54), (36, 241, 43, 68, 64),
                          (37, 223, 62, 56, 52), (38, 179, 71, 44, 44), (39, 11, 40, 58, 64), (40, 0, 68, 40, 44)])
    #imLink = r'F:\FH_Frankfurt\Python Adrian\Working Codes\Test_CNN\Data\stopSignDouble\007.jpg' 
    #image = cv2.imread(imLink)
    counter = 0
    counterTotal = 0
    #label = imLink.split(os.path.sep)[-1][0:3]
    for cropReg in croppingRegionList:
        (imName, x0, y0, imWidth, imHeight) = cropReg
        imLink = r'Data\stopSignDouble\{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(x0, y0, imWidth, imHeight), windowSize=(20,20), stepSize=(2,2)):
            counter += 1
            
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.95, 1.05)).astype('int')
            winArray = window.flatten()
            for i in range(0, len(winArray)):
                if winArray[i] > 255:
                    winArray[i] = 255
            winArray = winArray.astype('uint8')
            window = np.reshape(winArray, (20, 20, 3))
                
            cv2.imwrite(r'Data\stopSignDouble\croppedStopSignSections\{:03}{:03}{:03}.png'.format(imName, x, y), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of stop sign sections created is {}'.format(len(croppingRegionList), counterTotal))


'''
Cropping from 18x18 to 20x20, then resize to 20x20 sections that contains part of additional blue and misc background
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main02():
    croppingRegionList =  np.array([(3, 0, 50, 100, 100), (5, 0, 35, 100, 100), (5, 115, 40, 110, 80), (9, 90, 55, 140, 80),
                                    (9, 93, 58, 140, 80), (9, 93, 61, 140, 80), (9, 98, 58, 140, 80), (9, 98, 61, 140, 80),
                                    (13, 190, 55, 70, 70), (17, 50, 35, 30, 50), (19, 100, 75, 30, 60), (21, 140, 65, 70, 60),
                                    (23, 190, 85, 80, 50), (25, 155, 90, 160, 40), (27, 130, 80, 100, 50), 
                                    (35, 170, 75, 150, 60)
                                    ])
    
    counter = 0
    counterTotal = 0
    
    for cropReg in croppingRegionList:
        (imName, x0, y0, imWidth, imHeight) = cropReg
        imLink = r'Data\signError\signErr{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(x0, y0, imWidth, imHeight), windowSize=(20,20), stepSize=(10,10)):
            counter += 1
            
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.95, 1.05)).astype('int')
            winArray = window.flatten()
            for i in range(0, len(winArray)):
                if winArray[i] > 255:
                    winArray[i] = 255
            winArray = winArray.astype('uint8')
            window = np.reshape(winArray, (20, 20, 3))
            
            # Randomly change size of crop area
            decrement = random.randint(0, 2)
            if decrement != 0:
                window = window[1:(20-decrement+1), 1:(20-decrement+1)]
                window = cv2.resize(window, (20, 20), interpolation=cv2.INTER_AREA)
                
            cv2.imwrite(r'Data\stopSignDouble\croppedBackgroundSections\blue{:03}{:03}{:03}.png'.format(imName, x, y), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of background sections created is {}'.format(len(croppingRegionList), counterTotal))


'''
Cropping from 18x18 to 20x20, then resize to 20x20 sections that contains part of the blue background
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main01():
    croppingRegionList =  np.array([(52, 75, 60, 200, 180), (53, 190, 75, 80, 60)
                                    ])
    
    counter = 0
    counterTotal = 0
    
    for cropReg in croppingRegionList:
        (imName, x0, y0, imWidth, imHeight) = cropReg
        imLink = r'Data\stopSignDouble\blueBackground\blueBgr{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(x0, y0, imWidth, imHeight), windowSize=(20,20), stepSize=(10,10)):
            counter += 1
            
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.95, 1.05)).astype('int')
            winArray = window.flatten()
            for i in range(0, len(winArray)):
                if winArray[i] > 255:
                    winArray[i] = 255
            winArray = winArray.astype('uint8')
            window = np.reshape(winArray, (20, 20, 3))
            
            # Randomly change size of crop area
            decrement = random.randint(0, 2)
            if decrement != 0:
                window = window[1:(20-decrement+1), 1:(20-decrement+1)]
                window = cv2.resize(window, (20, 20), interpolation=cv2.INTER_AREA)
                
            cv2.imwrite(r'Data\stopSignDouble\croppedBackgroundSections\{:03}{:03}{:03}.png'.format(imName, x, y), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of background sections created is {}'.format(len(croppingRegionList), counterTotal))
    

'''
Cropping from 18x18 to 20x20, then resize to 20x20 sections that contains part of the red background
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main0():
    croppingRegionList =  np.array([(1, 240, 0, 80, 135), (8, 160, 0, 80, 135), (10, 80, 0, 80, 135), (12, 0, 0, 80, 135),
                                    (21,240, 0, 80, 135), (25, 160, 0, 80, 135), (31, 80, 0, 80, 135), (35, 0, 0, 80, 135),
                                    (41, 30, 29, 120, 144), (42, 28, 44, 100, 128), (43, 96, 87, 224, 90), (44, 195, 140, 124, 64),
                                    (45, 101, 109, 140, 100), (46, 119, 89, 152, 72), (47, 127, 99, 100, 100), (48, 179, 107, 140, 40),
                                    (50, 83, 12, 96, 100), (51, 70, 98, 120, 132)
                                    ])
    
    counter = 0
    counterTotal = 0
    
    for cropReg in croppingRegionList:
        (imName, x0, y0, imWidth, imHeight) = cropReg
        imLink = r'Data\stopSignDouble\{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(x0, y0, imWidth, imHeight), windowSize=(20,20), stepSize=(4,4)):
            counter += 1
            
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.95, 1.05)).astype('int')
            winArray = window.flatten()
            for i in range(0, len(winArray)):
                if winArray[i] > 255:
                    winArray[i] = 255
            winArray = winArray.astype('uint8')
            window = np.reshape(winArray, (20, 20, 3))
            
            # Randomly change size of crop area
            decrement = random.randint(0, 2)
            if decrement != 0:
                window = window[1:(20-decrement+1), 1:(20-decrement+1)]
                window = cv2.resize(window, (20, 20), interpolation=cv2.INTER_AREA)
                
            cv2.imwrite(r'Data\stopSignDouble\croppedBackgroundSections\{:03}{:03}{:03}.png'.format(imName, x, y), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of background sections created is {}'.format(len(croppingRegionList), counterTotal))


'''
Cropping from 18x18 to 20x20, then resize to 20x20 sections that contains part of the background
Format of croppingRegionList is (image name, x, y, width, height) where (x, y) is top left corner of the region,
width-height is the width and height of the region.
'''
def main0old():
    croppingRegionList =  np.array([(1, 240, 48, 80, 84), (8, 160, 48, 80, 84), (10, 80, 48, 80, 84), (12, 0, 48, 80, 84),
                                    (21,240, 48, 80, 84), (25, 160, 48, 80, 84), (31, 80, 48, 80, 84), (35, 0, 48, 80, 84)])
    
    counter = 0
    counterTotal = 0
    
    for cropReg in croppingRegionList:
        (imName, x0, y0, imWidth, imHeight) = cropReg
        imLink = r'Data\stopSignDouble\{:03}.jpg'.format(imName) 
        image = cv2.imread(imLink)
        for (x, y, window) in CNNFunc.slideWindow(image, ROI=(x0, y0, imWidth, imHeight), windowSize=(20,20), stepSize=(2,2)):
            counter += 1
            
            # Randomly change brightness of crop area
            window = (window * random.uniform(0.95, 1.05)).astype('int')
            winArray = window.flatten()
            for i in range(0, len(winArray)):
                if winArray[i] > 255:
                    winArray[i] = 255
            winArray = winArray.astype('uint8')
            window = np.reshape(winArray, (20, 20, 3))
            
            # Randomly change size of crop area
            decrement = random.randint(0, 2)
            if decrement != 0:
                window = window[1:(20-decrement+1), 1:(20-decrement+1)]
                window = cv2.resize(window, (20, 20), interpolation=cv2.INTER_AREA)
                
            cv2.imwrite(r'Data\stopSignDouble\croppedBackgroundSections\{:03}{:03}{:03}.png'.format(imName, x, y), window)
            
        counterTotal = counterTotal + counter
        print('Finished working on image {:03}.jpg, {} sections created.'.format(imName, counter))
        counter = 0
        
    print('Finish working on {} images, total number of background sections created is {}'.format(len(croppingRegionList), counterTotal))
    
if __name__ == '__main__': main02()