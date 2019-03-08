# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:23:58 2019

@author: Cheng Lin
"""

#Cheng Lin
#MAIS 202 Bootcamp final project
#ASL classifier training code (CNN)
#9 Mar 2019

from PIL import Image
import os
import sys
import cv2 as cv
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import pickle