import numpy as np
import cv2

stop_sign = cv2.CascadeClassifier('stop_sign.xml')
traffic_light = cv2.CascadeClassifier('traffic_light.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sign= stop_sign.detectMultiScale(gray, 1.3, 5)
    light = traffic_light.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in sign:
        cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
	print("stop sign")
    for (x,y,w,h) in light:
        cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
	print("taffic sign")

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
       break 


cap.release()
cv2.destroyAllWindows()
