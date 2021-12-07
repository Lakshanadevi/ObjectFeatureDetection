'''
To detect shape and color of an object in an image using openCV
Color - Red,Blue,Green,Yellow are detectable
Shape - Triangle,Square,Rectangle,Pentagon,Hexagon,Circle are detectable
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

from colorDetection.py import colorDetector
from shapeDetection.py import shapeDetector

def main(img_path):
    img = cv2.imread(img_path) #Read image
    
    #Create color detection object and detect color of the item in the image
    color_detector1 = colorDetector(img)
    color_detector1.create_color_mask()
    color_detector1.detect_colors()
    img_color = color_detector1.mark_colors()
    
    #Create shape detection object and detect shape of the item in the image
    shape_detector1 = shapeDetector(img)
    shape_detector1.preprocess_image()
    img_shape = shape_detector1.identify_shape()
    
    '''
    #Create a plot using matplotlib
    fig = plt.figure(figsize=(10, 7))
    rows = 1
    columns = 2
    
    img_color_rgb = cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_color_rgb)
    plt.axis('off')
    plt.title("Color Detection")

    img_shape_rgb = cv2.cvtColor(img_shape,cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_shape_rgb)
    plt.axis('off')
    plt.title("Shape Detection")
    '''
    cv2.imshow('Shape & Color', img_shape)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    '''
    Get the image path as an argument
    '''
    
    if len(sys.argv) == 0:
        print('Provide image path')
    elif len(sys.argv) >> 1:
        print('Provide only one image path')
    else:
        img_path = sys.argv
    
    main(img_path)
