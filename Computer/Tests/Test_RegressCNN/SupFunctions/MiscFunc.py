# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:41:43 2018

@author: tranl
"""
import os

def paths0(directory):
    fileNames = [f for f in os.listdir(directory)]
    filePaths = []
    for fileName in fileNames:
        filePath = '\\'.join((directory, fileName))
        filePaths.append(filePath)
    return filePaths

# This paths function assume that inside directory, there are folders with names 
# the same as classes and images of each class will be inside their respective folders
def paths(directory):
    classNames = [f for f in os.listdir(directory)]
    classPaths = []
    filePaths = []
    for className in classNames:
        classPath = '\\'.join((directory, className))
        classPaths.append(classPath)
    for classPath in classPaths:
        fileNames = [f for f in os.listdir(classPath)]
        for fileName in fileNames:
            filePath = '\\'.join((classPath, fileName))
            filePaths.append(filePath)
    return filePaths