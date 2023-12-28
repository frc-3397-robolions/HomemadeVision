#Error Key:
# >=0: No error
# -1: No contour found
# -2: Exception on coprocessor. Could be a camera error
# -3: Robot code didn't find a value on NetworkTables
# -4: Processing disabled

import cv2
import numpy as np
import json
from networktables import NetworkTables


#find the contours and draw them on the image
def findCenter(contours):
    if len(contours) == 0:
        return -1, -1
    M = cv2.moments(contours[0])
    cX = int(M['m10']/M['m00']) 
    cY = int(M['m01']/M['m00'])
    #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
    #cv2.imshow("Result", imgContour)
    return cX, cY
        

#filter out pixels not of a certain color
def hsv_filter(img):
    lower = np.array([h_min,s_min,v_min]) 
    upper = np.array([h_max,s_max,v_max])
    return cv2.inRange(img, lower, upper)


def find_edges(img):
    imgCanny = cv2.Canny(img, threshold1, threshold2)
    # kernel = np.ones((3,3), np.uint8)
    # imgErode = cv2.erode(imgCanny, kernel,iterations=1) 
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    return imgDil


#read config data from config.json
config = json.load(open('config.json'))

brightness = config["brightness"]
threshold1 = config["threshold1"] 
threshold2 = config["threshold2"]
h_min = config["h_min"]
h_max = config["h_max"]
s_min = config["s_min"] 
s_max = config["s_max"]
v_min = config["v_min"]
v_max = config["v_max"]
robot_ip = config["robot_ip"]
camera = config["camera"]
frameWidth = config["frameWidth"] 
frameHeight = config["frameHeight"]

cap = cv2.VideoCapture(camera)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

NetworkTables.initialize(server=robot_ip)
table = NetworkTables.getTable('vision')


#main loop
while True:
    #Check NT to see if should disable.
    if table.getBoolean("Vision Enabled",True) == False:
        table.putNumber('Target X', -4)
        table.putNumber('Target Y', -4)
        continue
    # Read the next frame from the camera
    success, img = cap.read()
    # Make a copy of the original image to draw contours on later
    try:
        # Adjust brightness of image
        img = cv2.convertScaleAbs(img, alpha=brightness/100)
    except:
        print("Error: Failed to read camera frame.")
        table.putNumber('Target X', -2)
        table.putNumber('Target Y', -2)
    # Blur the image to reduce noise
    img = cv2.GaussianBlur(img, (7, 7), 1)  
    # Apply color thresholding to extract objects of interest
    img = hsv_filter(img)  
    # Combine threshold mask with original image
    #img = cv2.bitwise_and(img, img, mask=filter)
    # Find and enhance edges in filtered image
    img = find_edges(img)

    # Detect contours in edge image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #find center of best contour
    targetX, targetY = findCenter(contours)
    table.putNumber('Target X', targetX)
    table.putNumber('Target Y', targetY)


    # Quit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        break