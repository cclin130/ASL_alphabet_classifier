# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:46:25 2019

@author: Cheng Lin
"""

#Cheng Lin
#MAIS 202 Bootcamp final project
#ASL classifier image preprocessing
#8 Mar 2019

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def fix_img_contrast(img_path):
    image = cv.imread(img_path)
    resized_image = cv.resize(image, (200,200))
    
    new_image = np.zeros(resized_image.shape, resized_image.dtype)
    
    alpha = 1.25 #contrast control
    beta = 0.5 #brightness control
    
    #we transform the image to obtain:
    # new_image(i,j) = alpha*image(i,j) + beta
    
    new_image = cv.convertScaleAbs(resized_image,alpha=alpha, beta=beta)

    #display results
    #plt.imshow(new_image)