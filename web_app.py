# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:22:44 2019

@author: Cheng Lin
"""

from cv2 import *
# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    namedWindow("cam-test")
    imshow("cam-test",img)
    waitKey(0)
    destroyWindow("cam-test")
    imwrite("filename.jpg",img) #save imagefrom SimpleCV import Image, Camera


from SimpleCV import Image, Camera

cam = Camera()
img = cam.getImage()
img.save("filename.jpg")