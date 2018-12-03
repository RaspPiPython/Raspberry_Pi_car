# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:59:10 2018

@author: tranl
"""

import socket
import pickle
import time

class PiImageClient:
    def __init__(self):
        self.s = None
        self.counter = 0
        
    def connectClient(self, serverIP, serverPort):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((serverIP, serverPort))
        
    def connectClientNODELAY(self, serverIP, serverPort):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.connect((serverIP, serverPort))
        
    def closeClient(self):
        self.s.close()
        
    def receiveOneImage(self):
        imageData = b''
        lenData = self.s.recv(8)
        length = pickle.loads(lenData) # should be 921764 for 640x480 images
        print('Data length is:', length)
        while len(imageData) < length:
            toRead = length-len(imageData)
            imageData += self.s.recv(4096 if toRead>4096 else toRead)
            #if len(imageData)%200000 <= 4096:
            #    print('Received: {} of {}'.format(len(imageData), length))
        return imageData
    
    def receiveFrame0(self):        
        imageData = b''
        lenData = self.s.recv(8) 
        length = pickle.loads(lenData)
        print('Data length is:', length)
        '''length = 921764 # for 640x480 images
        length = 230563 # for 320x240 images'''
        while len(imageData) < length:
            toRead = length-len(imageData)
            imageData += self.s.recv(4096 if toRead>4096 else toRead)
            #if len(imageData)%200000 <= 4096:
            #    print('Received: {} of {}'.format(len(imageData), length))
        self.counter += 1
        if len(imageData) == length: 
            print('Successfully received frame {}'.format(self.counter))                
        return imageData
    
    def receiveFrame(self):        
        imageData = b''
        lenString = self.s.recv(6) 
        length = int(lenString)
        print('Data length is:', length)
        while len(imageData) < length:
            toRead = length-len(imageData)
            imageData += self.s.recv(4096 if toRead>4096 else toRead)
            #if len(imageData)%200000 <= 4096:
            #    print('Received: {} of {}'.format(len(imageData), length))
        self.counter += 1
        if len(imageData) == length: 
            print('Successfully received frame {}'.format(self.counter))                
        return imageData
    
    def sendCommand(self, command):
        #if len(command) != 3:
        #    print('<WARNING> Length of command string is different from 3')
        
        self.s.send(command.encode())
        #commandData = pickle.dumps(command)
        #self.s.send(commandData)
        print('Command {} sent'.format(command))
        
    def receiveCommand(self):
        commandData = self.s.recv(3)
        command = commandData.decode()
        return command
        
    
class PiImageServer:
    def __init__(self):
        self.s = None
        self.conn = None
        self.addr = None
        #self.currentTime = time.time()
        self.currentTime = time.asctime(time.localtime(time.time()))
        self.counter = 0

    def openServer(self, serverIP, serverPort):
        print('<INFO> Opening image server at {}:{}'.format(serverIP,
                                                            serverPort))
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((serverIP, serverPort))
        self.s.listen(1)
        print('Waiting for client...')
        self.conn, self.addr = self.s.accept()
        print('Connected by', self.addr)

    def closeServer(self):
        print('<INFO> Closing server...')
        self.conn.close()
        self.s.close()
        #self.currentTime = time.time()
        self.currentTime = time.asctime(time.localtime(time.time()))
        print('Server closed at', self.currentTime)

    def sendOneImage(self, imageData):
        print('<INFO> Sending only one image...')
        imageDataLen = len(imageData)
        lenData = pickle.dumps(imageDataLen)
        print('Sending image length')
        self.conn.send(lenData)
        print('Sending image data')
        self.conn.send(imageData)

    def sendFrame(self, frameData):
        self.counter += 1
        print('Sending frame ', self.counter)
        frameDataLen = len(frameData)
        lenData = pickle.dumps(frameDataLen)        
        self.conn.send(lenData)        
        self.conn.send(frameData)
        
    def recvCommand(self):
        commandData = self.conn.recv(3)
        command = commandData.decode()
        return command
        