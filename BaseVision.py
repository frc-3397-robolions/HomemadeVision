import cv2
import numpy as np
import json
import time

frameWidth = 640
frameHeight = 480
brightness=0.5

cap = cv2.VideoCapture(0)

cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 340)

#read saved config data in config.json
def read_config():
    with open('config.json') as f:
      config = json.load(f)

    threshold1 = config["threshold1"] 
    threshold2 = config["threshold2"]
    brightness = config["brightness"]
    h_min = config["h_min"]
    h_max = config["h_max"] 
    s_min = config["s_min"]
    s_max = config["s_max"]
    v_min = config["v_min"]
    v_max = config["v_max"]
    return brightness,threshold1,threshold2,h_min,h_max,s_min,s_max,v_min,v_max


#create trackbars for adjusting the parameters
def createTrackbars(brightness, empty, threshold1, threshold2, h_min, h_max, s_min, s_max, v_min, v_max):
    cv2.createTrackbar("Threshold1", "Parameters", threshold1, 10000, empty)
    cv2.createTrackbar("Threshold2", "Parameters", threshold2, 10000, empty) 
    cv2.createTrackbar('Brightness', 'Parameters', brightness, 100, empty)
    cv2.createTrackbar('H Min', 'Parameters', h_min, 255, empty)
    cv2.createTrackbar('H Max', 'Parameters', h_max, 255, empty) 
    cv2.createTrackbar('S Min', 'Parameters', s_min, 255, empty)
    cv2.createTrackbar('S Max', 'Parameters', s_max, 255, empty)
    cv2.createTrackbar('V Min', 'Parameters', v_min, 255, empty)
    cv2.createTrackbar('V Max', 'Parameters', v_max, 255, empty)

#find the contours and draw them on the image
def getContours(img, imgContour):
    #try cv2.RETR_TREE?
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000:
            para = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*para, True)
            x,y,w,h = cv2.boundingRect(approx)
            if len(approx)>20:
                continue
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 3)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(imgContour, "Points: "+str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: "+str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

#filter out pixels not of a certain color
def hsv_filter(img):
    h_min = cv2.getTrackbarPos('H Min', 'Parameters')
    h_max = cv2.getTrackbarPos('H Max', 'Parameters')
    s_min = cv2.getTrackbarPos('S Min', 'Parameters') 
    s_max = cv2.getTrackbarPos('S Max', 'Parameters')
    v_min = cv2.getTrackbarPos('V Min', 'Parameters')
    v_max = cv2.getTrackbarPos('V Max', 'Parameters')

    lower = np.array([h_min,s_min,v_min]) 
    upper = np.array([h_max,s_max,v_max])
    return cv2.inRange(img, lower, upper)


def find_edges(img):
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(img, threshold1, threshold2)
    # kernel = np.ones((3,3), np.uint8)
    # imgErode = cv2.erode(imgCanny, kernel,iterations=1) 
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    return imgDil


#read config data from config.json
brightness, threshold1, threshold2, h_min, h_max, s_min, s_max, v_min, v_max = read_config()
#use config data as defaults for trackbars
createTrackbars(brightness, empty, threshold1, threshold2, h_min, h_max, s_min, s_max, v_min, v_max)

#main loop
while True:
    start = time.time()  
    # Read the next frame from the camera
    success, img = cap.read()

    # Make a copy of the original image to draw contours on later
    imgContour = img.copy()

    # Adjust brightness of image
    imgBright = cv2.convertScaleAbs(img, alpha=cv2.getTrackbarPos('Brightness', 'Parameters')/100)

    # Blur the image to reduce noise
    imgBlur = cv2.GaussianBlur(imgBright, (7, 7), 1)  

    # Apply color thresholding to extract objects of interest
    filter = hsv_filter(imgBright)  

    # Combine threshold mask with original image
    filter = cv2.bitwise_and(imgBright, imgBright, mask=filter)

    # Find and enhance edges in filtered image
    imgDil = find_edges(filter)

    # Detect contours in edge image
    getContours(imgDil, imgContour)

    # Display the filtered image
    cv2.imshow("Result", filter)

    # Display the image with contours drawn
    cv2.imshow("Contours", imgContour)

    # Quit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        with open('config.json') as f:
            config = json.load(f)

        config["threshold1"] = cv2.getTrackbarPos("Threshold1", "Parameters")
        config["threshold2"] = cv2.getTrackbarPos("Threshold2", "Parameters")
        config["brightness"] = cv2.getTrackbarPos("Brightness", "Parameters")
        config["h_min"] = cv2.getTrackbarPos("H Min", "Parameters")
        config["h_max"] = cv2.getTrackbarPos("H Max", "Parameters")
        config["s_min"] = cv2.getTrackbarPos("S Min", "Parameters") 
        config["s_max"] = cv2.getTrackbarPos("S Max", "Parameters")
        config["v_min"] = cv2.getTrackbarPos("V Min", "Parameters")
        config["v_max"] = cv2.getTrackbarPos("V Max", "Parameters")
# etc

        with open('config.json', 'w') as f:
            json.dump(config, f)
        
        cv2.destroyAllWindows()
        cap.release()
        break
    
    end = time.time()
    duration = end - start
    fps = 1/duration
    print(f"FPS: {fps}")