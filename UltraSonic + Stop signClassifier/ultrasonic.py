from socket import *
import time
import RPi.GPIO as GPIO


GPIO.setwarnings(False)

def measure():
    """
    measure distance
    """
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
    start = time.time()

    while GPIO.input(GPIO_ECHO)==0:
        start = time.time()

    while GPIO.input(GPIO_ECHO)==1:
        stop = time.time()

    elapsed = stop-start
    distance = (elapsed * 34300)/2

    return distance

# referring to the pins by GPIO numbers
GPIO.setmode(GPIO.BOARD)

# define pi GPIO
GPIO_TRIGGER = 31
GPIO_ECHO    = 32

# output pin: Trigger
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)
# input pin: Echo
GPIO.setup(GPIO_ECHO,GPIO.IN)
# initialize trigger pin to low
GPIO.output(GPIO_TRIGGER, False)

try:
    while True:
        distance = measure()
        if distance <= 50:
            print "Distance : %.1f cm" % distance
            time.sleep(0.5)
finally:
    GPIO.cleanup()