from tkinter import *
import tkinter as tk
import time

class Application(Frame):
        def __init__(self, master=None):
                Frame.__init__(self, master)
                self.grid()
                self.createWidgets()

        def createWidgets(self):

                self.cnvs =  Canvas (bg='red', height=100, width=100)
                self.cnvs.grid(row=4, column=1, sticky=N)
                self.newbtn = Button ( self, text="Draw Line", command=hello_world)
                self.newbtn.grid(row=3, column=1)

def hello_world():
        print('HELLO WORLD')

def counter():
    i = 0
    while i<10:
        time.sleep(0.5)
        print(i)
        i+=1
        if flag.get() == True:
            i+=1
        if CheckVar1.get() == True:
            i+=1
        if CheckVar2.get() == True:
            i+=1
            

def toggleText():
    if button2['text'] == 'On':
        button2['text'] = 'Off'
        flag.set(False)
    else:
        button2['text'] = 'On'
        flag.set(True)

def printCheck():
    print(CheckVar1.get(), CheckVar2.get())


#app = Application()
#app.master.title("Sample application")
master = tk.Tk()
flag = tk.BooleanVar()
CheckVar1 = tk.BooleanVar()
CheckVar2 = tk.BooleanVar()
flag.set(False)
checkFlags = [CheckVar1, CheckVar2]
master.title('Sample loop')

button = tk.Button(master, text='Start', command=counter)
button.pack()
button2 = tk.Button(text='Off', command =toggleText)
button2.pack()
button3 = tk.Button(text='Print Check', command =printCheck)
button3.pack()
button4 = tk.Button(text='Exit', command =master.destroy)
button4.pack()

C1 = tk.Checkbutton(master, text = "Check1", variable = CheckVar1, \
                 onvalue = True, offvalue = False, height=5, \
                 width = 20)
C2 = tk.Checkbutton(master, text = "Check2", variable = CheckVar2, \
                 onvalue = True, offvalue = False, height=5, \
                 width = 20)
C1.pack()
C2.pack()

master.mainloop()