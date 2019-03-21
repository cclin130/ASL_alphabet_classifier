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
    img = cv.imread(img_path)
    resized_image = cv.resize(img, (200,200))
    
    new_image = np.zeros(resized_image.shape, resized_image.dtype)
    
    alpha = 1.25 #contrast control
    beta = 0.5 #brightness control
    
    #we transform the image to obtain:
    # new_image(i,j) = alpha*image(i,j) + beta
    
    new_image = cv.convertScaleAbs(resized_image,alpha=alpha, beta=beta)
    
    return resized_image


def img_threshold(img_path):
    img = cv.imread(img_path,0)
    resized_image = cv.resize(img, (200,200))
    
    #ret,img_thresh = cv.threshold(resized_image,127,255,cv.THRESH_TRUNC)
    img_thresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)    
    return img_thresh
    #plt.imshow(img, 'gray')