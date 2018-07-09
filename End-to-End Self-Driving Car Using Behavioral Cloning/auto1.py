# Import required modules
import time
import pygame
import RPi.GPIO as GPIO
from pygame.locals import *
import numpy as np
import io
import sys
import cv2 as cv
import picamera
import picamera.array
from keras.models import load_model
from scipy.misc import  imresize


# Define Pins
print ("start")
# Definr Pin
m11=35
m12=36
m21=37
m22=38
PWM= 80
PWM_L= 40


# Disable Warnings
GPIO.setwarnings(False)

class CarControl:
  def __init__(self):
    # Declare the GPIO settings
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(m11, GPIO.OUT)
    GPIO.setup(m12, GPIO.OUT)
    GPIO.setup(m21, GPIO.OUT)
    GPIO.setup(m22, GPIO.OUT)

    self.m1= GPIO.PWM(m11,1000)
    self.m2= GPIO.PWM(m12,1000)
    self.m3= GPIO.PWM(m21,1000)
    self.m4= GPIO.PWM(m22,1000)

    self.m1.start(0)
    self.m2.start(0)
    self.m3.start(0)
    self.m4.start(0)
    self.steer()


  def forward(self):
    self.m1.ChangeDutyCycle(PWM)
    self.m2.ChangeDutyCycle(0)
    self.m3.ChangeDutyCycle(PWM)
    self.m4.ChangeDutyCycle(0)

  def right(self):
    self.m1.ChangeDutyCycle(PWM_L)
    self.m2.ChangeDutyCycle(0)
    self.m3.ChangeDutyCycle(0)
    self.m4.ChangeDutyCycle(PWM_L)

  def left(self):
    self.m1.ChangeDutyCycle(0)
    self.m2.ChangeDutyCycle(PWM_L)
    self.m3.ChangeDutyCycle(PWM_L)
    self.m4.ChangeDutyCycle(0)

  def forwardRight(self):
     self.m1.ChangeDutyCycle(PWM)
     self.m2.ChangeDutyCycle(0)
     self.m3.ChangeDutyCycle(0)
     self.m4.ChangeDutyCycle(PWM_L)

  def forwardLeft(self):
     self.m1.ChangeDutyCycle(0)
     self.m2.ChangeDutyCycle(PWM)
     self.m3.ChangeDutyCycle(PWM_L)
     self.m4.ChangeDutyCycle(0)

  def stop(self):
    self.m1.ChangeDutyCycle(0)
    self.m2.ChangeDutyCycle(0)
    self.m3.ChangeDutyCycle(0)
    self.m4.ChangeDutyCycle(0)

  def reverse(self):
    self.m1.ChangeDutyCycle(0)
    self.m2.ChangeDutyCycle(PWM)
    self.m3.ChangeDutyCycle(0)
    self.m4.ChangeDutyCycle(PWM)

  # def reverseRight(self):
  #      self.m1.ChangeDutyCycle(0)
  #      self.m2.ChangeDutyCycle(PWM_L)
  #      self.m3.ChangeDutyCycle(PWM)
  #      self.m4.ChangeDutyCycle(0)

  # def reverseLeft(self):
  #      self.m1.ChangeDutyCycle(PWM_L)
  #      self.m2.ChangeDutyCycle(0)
  #      self.m3.ChangeDutyCycle(0)
  #      self.m4.ChangeDutyCycle(PWM)

# 1 Forward
# 2 Right
# 3 Left
# 4 forwardRight
# 5 forwardLeft
# 6 reverse

  def steer(self):
    #print(1)
    loaded_model = load_model('123.hdf5')
    #print(1.1)
    with picamera.PiCamera() as camera:
      with picamera.array.PiRGBArray(camera) as stream:
        #print(1.2)
        camera.resolution = (320, 240)      # pi camera resolution
        camera.framerate = 15
        camera.start_preview()
        time.sleep(5)
        # save filename + ' ' + command value 0,1,..etc in line in file called data.txt
        #listen for pygame event and update command
        #if command is exit break
        #modify to listen for pygame
        # stream = io.BytesIO()
        starttime = time.time()
        print(2)
        for foo in camera.capture_continuous(stream, 'rgb', use_video_port=True):
          # while self.send_inst:
          data = stream.array
          data = data[120:,:]
          data = imresize(data, size=(320, 120))
          #print(data.shape)
          image = np.expand_dims(np.float32(data) / 255, axis=0)
          command = np.argsort(loaded_model.predict(image)[0])[-1] + 1
          #command = 0
          #print(command)
          if command == 5:
            # print("forwardRight")
            self.forwardRight()

          elif command == 6:
            # print("forwardLeft")
            self.forwardLeft()

          elif command == 1:
            #print("Forward")
            self.forward()

          elif command == 2:
            #print("Right")
            self.right()

          elif command == 3:
            #print("Left")
            self.left()

          elif command == 4:
            # print("Reverse")
            self.reverse()
          #print(3)

          # elif command == 4:
          #      # print("reverseRight")
          #      self.reverseRight()

          # elif command == 6:
          #      # print("reverseLeft")
          #      self.reverseLeft()
          stream.seek(0)

if __name__ == "__main__":
    CarControl()
