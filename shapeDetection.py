'''
Detect shape of an object in a given image.
Traingle, square, rectangle, pentagon, hexagon and circle can be detected
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

class shapeDetector():
    def __init__(self, img):
        self.img = img
    
    def preprocess_image(self):
        '''
        Process the image in order to obtain useful corners and edges without noise
        '''
        
        img_blurred = cv2.GaussianBlur(self.img,(5,5),0) #Gaussian blur to remove noise
        img_gray = cv2.cvtColor(img_blurred,cv2.COLOR_BGR2GRAY) #edge detection in a gray scale image is more effective
        
        _, threshold_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        
        threshold1 = 30
        threshold2 = 35
        img_canny = cv2.Canny(img_gray,threshold1,threshold2) #detect the edges and corners of the object in the image, thresholds eliminate noise
        
        kernel = np.ones((5,5))
        self.img_dilation = cv2.dilate(img_canny, kernel, iterations=1) #dilation helps in increasing the confidence of object boundry
    
    def identify_shape(self):
        '''
        use contours to mark the object boundaries. Based on the contours and number of line segments, identify the shape.
        '''
        contours,hierarchy = cv2.findContours(self.img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            #cv2.drawContours(self.img, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])


            # putting shape name at center of each shape
            if len(approx) == 3:
                cv2.putText(self.img, 'Triangle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif len(approx) == 4:
                (x1, y1, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                # aspect ratio of square is approximately 1
                if ar >= 0.95 and ar <= 1.05:
                    cv2.putText(self.img, 'Square', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                else:
                    cv2.putText(self.img, 'Rectangle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif len(approx) == 5:
                cv2.putText(self.img, 'Pentagon', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            elif len(approx) == 6:
                cv2.putText(self.img, 'Hexagon', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            else:
                cv2.putText(self.img, 'circle', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
        return self.img
