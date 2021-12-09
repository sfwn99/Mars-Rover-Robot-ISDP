import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class Motor():
    def __init__(self,Ena1,In1,In2,Ena2,In3,In4):
        self.Ena1 = Ena1
        self.In1 = In1
        self.In2 = In2
        self.Ena2 = Ena2
        self.In3 = In3
        self.In4 = In4

        GPIO.setup(self.Ena1,GPIO.OUT)
        GPIO.setup(self.In1,GPIO.OUT)
        GPIO.setup(self.In2,GPIO.OUT)
        GPIO.setup(self.Ena2,GPIO.OUT)
        GPIO.setup(self.In3,GPIO.OUT)
        GPIO.setup(self.In4,GPIO.OUT)
        
        self.pwm1 = GPIO.PWM(self.Ena1,100)
        self.pwm1.start(0)
        self.pwm2 = GPIO.PWM(self.Ena2,100)
        self.pwm2.start(0)
        
    def moveB(self,x=50,t=1):
        GPIO.output(self.In1,GPIO.LOW)
        GPIO.output(self.In2,GPIO.HIGH)
        GPIO.output(self.In3,GPIO.LOW)
        GPIO.output(self.In4,GPIO.HIGH)
        self.pwm1.ChangeDutyCycle(x)
        self.pwm2.ChangeDutyCycle(x)
        sleep(t)
        
    def moveF(self,x=50,t=1):
        GPIO.output(self.In1,GPIO.HIGH)
        GPIO.output(self.In2,GPIO.LOW)
        GPIO.output(self.In3,GPIO.HIGH)
        GPIO.output(self.In4,GPIO.LOW)
        self.pwm1.ChangeDutyCycle(x)
        self.pwm2.ChangeDutyCycle(x)
        sleep(t)
    
    def rotr(self,x=50,t=1):
        GPIO.output(self.In1,GPIO.LOW)
        GPIO.output(self.In2,GPIO.HIGH)
        GPIO.output(self.In3,GPIO.HIGH)
        GPIO.output(self.In4,GPIO.LOW)
        self.pwm1.ChangeDutyCycle(x)
        self.pwm2.ChangeDutyCycle(x)
        sleep(t)
        
    def rotl(self,x=50,t=1):
        GPIO.output(self.In1,GPIO.HIGH)
        GPIO.output(self.In2,GPIO.LOW)
        GPIO.output(self.In3,GPIO.LOW)
        GPIO.output(self.In4,GPIO.HIGH)
        self.pwm1.ChangeDutyCycle(x)
        self.pwm2.ChangeDutyCycle(x)
        sleep(t)
    
    def stop(self,t=0):
        self.pwm1.ChangeDutyCycle(0)
        self.pwm2.ChangeDutyCycle(0)
        sleep(t)

class Servo():
    def __init__(self,pin):
        self.pin = pin
        GPIO.setup(self.pin,GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin,50)
        self.pwm.start(0)
        
    def opens(self):
        # Open claw
        self.pwm.ChangeDutyCycle(12)
        sleep(5)
    def closes(self):
        print("Closing Claw")
        #Grip claw
        self.pwm.ChangeDutyCycle(2)
        sleep(10)
        #Release
        self.pwm.ChangeDutyCycle(0)
        sleep(30)
        

servo = Servo(14)
motor = Motor(2,3,4,17,27,22)


    
    
    

