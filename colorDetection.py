'''
Detect color of an object in a given image.
Red, Blue, Green and Yelow can be detected
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

class colorDetector:
    def __init__(self, img):
        self.img = img
    
    def create_color_mask(self):
        '''
        Create a mask for each color to identify the colors in an image
        '''
        #Convert to hsv format which provides unique values for each color
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        #Obtain lower and upper range for each color using trial and error or previous works
        #create a color mask in order to identify the color in any part of the image
        blue_lower_range = np.array([94,80,2])
        blue_upper_range = np.array([120,255,255])    
        self.blue_mask = cv2.inRange(img_hsv, blue_lower_range, blue_upper_range)
        
        red_lower_range = np.array([136,87,111])
        red_upper_range = np.array([180,255,255])
        self.red_mask = cv2.inRange(img_hsv, red_lower_range, red_upper_range)
        
        green_lower_range = np.array([45,52,72])
        green_upper_range = np.array([93,255,255])
        self.green_mask = cv2.inRange(img_hsv, green_lower_range, green_upper_range)
        
        yellow_lower_range = np.array([22,93,0])
        yellow_upper_range = np.array([45,255,255])
        self.yellow_mask = cv2.inRange(img_hsv, yellow_lower_range, yellow_upper_range)
    
    def detect_colors(self):
        '''
        Detect each color in an image individually using color masks
        '''
        kernal = np.ones((5,5), "uint8")
        
        self.red_mask = cv2.dilate(self.red_mask, kernal)
        self.red_res = cv2.bitwise_and(self.img, self.img, mask=self.red_mask)
        
        self.green_mask = cv2.dilate(self.green_mask, kernal)
        self.green_res = cv2.bitwise_and(self.img, self.img, mask=self.green_mask)
        
        self.blue_mask = cv2.dilate(self.blue_mask, kernal)
        self.blue_res = cv2.bitwise_and(self.img, self.img, mask=self.blue_mask)
        
        self.yellow_mask = cv2.dilate(self.yellow_mask, kernal)
        self.yellow_res = cv2.bitwise_and(self.img, self.img, mask=self.yellow_mask)
    
    def mark_colors(self):
        '''
        Use contours to mark the color regions and write detected color name in the original image
        '''
        contours_col, hierarchy_col = cv2.findContours(self.red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cnt in enumerate(contours_col):
            area = cv2.contourArea(cnt)
            if area > 300:
                x,y,w,h = cv2.boundingRect(cnt)
                self.img = cv2.rectangle(self.img, (x,y),(x+w, y+h), (0,0,255),2)
                cv2.putText(self.img, "Red", (x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
        
        contours_col, hierarchy_col = cv2.findContours(self.green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cnt in enumerate(contours_col):
            area = cv2.contourArea(cnt)
            if area > 300:
                x,y,w,h = cv2.boundingRect(cnt)
                self.img = cv2.rectangle(self.img, (x,y),(x+w, y+h), (0,0,255),2)
                cv2.putText(self.img, "Green", (x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
        
        contours_col, hierarchy_col = cv2.findContours(self.blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cnt in enumerate(contours_col):
            area = cv2.contourArea(cnt)
            if area > 300:
                x,y,w,h = cv2.boundingRect(cnt)
                self.img = cv2.rectangle(self.img, (x,y),(x+w, y+h), (0,0,255),2)
                cv2.putText(self.img, "Blue", (x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
        
        contours_col, hierarchy_col = cv2.findContours(self.yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, cnt in enumerate(contours_col):
            area = cv2.contourArea(cnt)
            if area > 300:
                x,y,w,h = cv2.boundingRect(cnt)
                self.img = cv2.rectangle(self.img, (x,y),(x+w, y+h), (0,0,255),2)
                cv2.putText(self.img, "Yellow", (x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255))
        
        return self.img
