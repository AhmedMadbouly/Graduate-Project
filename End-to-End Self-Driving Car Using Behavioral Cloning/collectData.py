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

# Define Pin
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


          # Setup and begin pygame
          pygame.init()

          pygame.display.set_mode((400, 300))
          self.send_inst = True
          self.steer()


     def forward(self):
          self.m1.ChangeDutyCycle(PWM)
          self.m2.ChangeDutyCycle(0)
          self.m3.ChangeDutyCycle(PWM)
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
          self.m2.ChangeDutyCycle(PWM_L)
          self.m3.ChangeDutyCycle(0)
          self.m4.ChangeDutyCycle(PWM_L)

# 1 Forward
# 2 forwardRight
# 3 forwardLeft
# 4 reverse




     def steer(self):
          #complex_cmd = False
          #for write to file
          with open('data.txt', 'a') as the_file:
               with picamera.PiCamera() as camera:
                    command = 0 #forward
                    camera.resolution = (160, 240)      # pi camera resolution
                    camera.framerate = 30
                    camera.start_preview()
                    time.sleep(5)
                    print ("start")
                    # save filename + ' ' + command value 0,1,..etc in line in file called data.txt
                    #listen for pygame event and update command
                    #if command is exit break
                    #modify to listen for pygame
                    frame = 0
                    stream = io.BytesIO()
                    starttime = time.time()
                    framecount = 0
                    for foo in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                         framecount+=1
                    # while self.send_inst:
                         for event in pygame.event.get():
                              if event.type == KEYDOWN:
                                   key_input = pygame.key.get_pressed()
                                   if key_input[pygame.K_w] and key_input[pygame.K_d]:
                                        command = 2
                                        # #print("forwardRight")
                                        self.forwardRight()

                                   elif key_input[pygame.K_w] and key_input[pygame.K_a]:
                                        command = 3
                                        #print("forwardLeft")
                                        self.forwardLeft()

                                   elif key_input[pygame.K_w]:
                                        command = 1
                                        #print("Forward")
                                        self.forward()

                                   elif key_input[pygame.K_s]:
                                        command = 4
                                        #print("Reverse")
                                        self.reverse()

                                   elif key_input[pygame.K_q]:
                                        command = 0
                                        self.send_inst = False
                                        #print("Exit")
                                        self.stop()
                                        break

                              elif event.type == pygame.KEYUP:
                                   command = 0
                                   self.stop()


                         if command != 0:
                              # #print("save")
                              frame += 1
                              # Construct a numpy array from the stream
                              data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                              # "Decode" the image from the array, preserving colour
                              image = cv.imdecode(data, 1)
                              cv.imwrite('collected_images/' + str(command-1) + '/frame{:>05}.jpg'.format(frame),image)
                              the_file.write('frame{:>05}.jpg'.format(frame) + ' ' + str(command) + '\n')
                         stream.seek(0)
                         if not self.send_inst:
                              break



if __name__ == "__main__":
    CarControl()
