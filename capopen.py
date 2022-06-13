import numpy as np
import cv2
print("running")
cap = cv2.VideoCapture("udpsrc port=53262 ! application/x-rtp, payload=96, clock-rate=90000 ! rtph264depay ! h264parse ! decodebin ! videoconvert  !  appsink", cv2.CAP_GSTREAMER)
while True:

	print("running") 
	print(cap)
	ret,frame=cap.read()
	print(ret,frame)
	if not ret:
		continue
	print("gelukt")
	cv2.imshow("receive",frame)
	
	if cv2.waitKey(1)&0xFF == ord('q'):
		break
