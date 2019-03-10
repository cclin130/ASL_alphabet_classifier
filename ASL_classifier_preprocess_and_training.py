# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Cheng Lin
#MAIS 202 Bootcamp, deliverable 2
#ASL classifier training code
#25 Feb 2019

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

#define function to read in image and vectorize using HOG
def img_to_HOG(file_name):
    #convert image to grayscale, then find hog using skimage library
    img = Image.open(file_name).convert('LA')
    #feature, visual = hog(img, visualize=True)
    img_hog = hog(img)
    return img_hog

def img_to_HOG_contrast(file_name):
    img = cv.imread(file_name)
    resized_image = cv.resize(img, (200,200))
    
    new_image = np.zeros(resized_image.shape, resized_image.dtype)
    
    alpha = 3.0 #contrast control
    beta = -175 #brightness control
    
    #we transform the image to obtain:
    # new_image(i,j) = alpha*image(i,j) + beta
    
    new_image = cv.convertScaleAbs(resized_image,alpha=alpha, beta=beta)
    return hog(new_image)

#read in all data, create a dataframe of data
if __name__ == '__main__':
    
    data = []
    label = []

    #loop through all folders
    corpus_path = 'data/asl_alphabet_train'
    for folder in os.listdir(corpus_path):
        full_path = os.path.join(corpus_path, folder)
        count=0
        for image in os.listdir(full_path):
            count+=1;
            print(image)
            image_path = os.path.join(full_path, image)
            temp = img_to_HOG(image_path)

            data.append(temp)
            label.append(folder)

            if count>=100: break

    #convert data to np array
    X = np.array(data)

    #convert labels to array and one-hot encode
    y_labels = np.array(label)

    le = LabelEncoder()
    le.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',\
         'X', 'Y', 'Z'])

    y = le.transform(y_labels)

    #get train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, random_state=42)

    #train model on data
    model = OneVsRestClassifier(svm.SVC(kernel='linear', verbose=True))
    model.fit(X_train, y_train)

    #calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    #confusion matrix
    mat = confusion_matrix(y_test, y_pred)

    sns.heatmap(mat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')

    plt.show()
    
    #save model
    with open('obj/model_test.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
        
    #load model:
    with open('obj/model1000.pkl','rb') as input:
        model = pickle.load(input)