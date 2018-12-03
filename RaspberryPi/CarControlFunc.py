import RPi.GPIO as IO
import time

class PiCarCtrl:
    def __init__(self):
        # initialize GPIO pins, forwardFlag and speed values
        IO.setwarnings(False)
        IO.setmode(IO.BCM)
        IO.setup(19,IO.OUT)#PWM channel 1
        IO.setup(18,IO.OUT)#PWM channel 0
        IO.setup(17,IO.OUT)#Motor direction control (IN1-IN3 on L298N board)
        IO.setup(27,IO.OUT)#Motor direction control (IN2-IN4 on L298N board)
        #Channel 0 is for forward-backward, channel 1 is for left-right
        #Channel 0 is at GPIO 12/18; Channel 1 is at GPIO 13/19
        #Frequency of PWM signal is 50Hz
        self.p0=IO.PWM(18,50)#PWM channel 0 (motor)
        self.p1=IO.PWM(19,50)#PWM channel 1 (servo motor)
        self.forwardFlag = True
        self.speedStraight = 50
        self.speedCurve = 45
        
        # start PWM channels
        self.motor = 0
        self.servo = 6.8
        self.p0.start(self.motor)#duty cycle = 0 (no power for motor)
        self.p1.start(self.servo)#duty cycle = 6.8 is the middle position of servo
        
    def forward(self):
        print("Forward")
        self.forwardFlag = True
        self.motor = self.speedStraight
        IO.output(17,False)
        IO.output(27,True)
        if self.servo != 6.8:         
            self.servo = 6.8
        self.p0.ChangeDutyCycle(self.motor)

    def backward(self):
        print("Backward")
        self.forwardFlag = False
        self.motor = self.speedStraight
        IO.output(17,True)
        IO.output(27,False)
        if self.servo != 6.8:
            self.servo = 6.8
        self.p0.ChangeDutyCycle(self.motor)
    def left(self):
        print("Left")        
        #self.servo = 5.6
        self.servo = 5.9
        self.motor = self.speedCurve
        self.p1.ChangeDutyCycle(self.servo)
        self.p0.ChangeDutyCycle(self.motor)

    def right(self):
        print("Right")
        #self.servo = 8.4
        self.servo = 8.2
        self.motor = self.speedCurve
        self.p1.ChangeDutyCycle(self.servo)
        self.p0.ChangeDutyCycle(self.motor)

    def brake(self):
        print("Braking")
        if self.forwardFlag == True:
            IO.output(17,True)
            IO.output(27,False)
        else:
            IO.output(17,False)
            IO.output(27,True)
        time.sleep(0.2)
        self.motor = 0
        self.servo = 6.8
        IO.output(17,False)
        IO.output(27,False)
        self.p0.ChangeDutyCycle(self.motor)
        self.p1.ChangeDutyCycle(self.servo)
    
    @staticmethod
    def plus(data):
        data = data + 1
        return data
    
    @staticmethod
    def plus2(data):
        data = CarCtrl.plus(data)
        data = data + 1
        return data

def main():
    i = 1
    print(CarCtrl.plus(i))
    print(CarCtrl.plus2(i))

if __name__ == '__main__': main()
