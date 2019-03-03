#Cheng Lin
#MAIS 202 Bootcamp, deliverable 2
#ASL classifier training code
#25 Feb 2019

from PIL import Image
import os
import sys
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

#read in all data, create a dataframe of data
if __name__ == '__main__':
    corpus_path = sys.argv[1]
    model_path = sys.argv[2]
    idx_start = int(sys.argv[3])
    idx_stop = int(sys.argv[4])
    
    data = []
    label = []

    #loop through all folders
    #corpus_path = data_dir
    print('----------retrieving data from ' + corpus_path + ' ---------------')
    for folder in os.listdir(corpus_path):
        full_path = os.path.join(corpus_path, folder)
        count=0
        for image in os.listdir(full_path):
            count+=1;
            print(image)
            
            if count >= idx_start and count < idx_stop:
                image_path = os.path.join(full_path, image)
            
                temp = img_to_HOG(image_path)
    
                data.append(temp)
                label.append(folder)

            if count>= idx_stop: break

    #convert data to np array
    X = np.array(data)

    #convert labels to array and one-hot encode
    y_labels = np.array(label)
    
    print('--------------encoding labels----------------')
    le = LabelEncoder()
    le.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
         'X', 'Y'])

    y = le.fit_transform(y_labels)

    #get train-test split
    print('----------------splitting data-----------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True, random_state=42)

    #train model on data
    
    
    print('----------------training model-----------------')
    model = OneVsRestClassifier(svm.SVC(kernel='linear', verbose=True))
    model.fit(X_train, y_train)

    #calculate accuracy
    print('------------------testing on validation set-------------')
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy for model :')
    print(accuracy)

    #confusion matrix
    mat = confusion_matrix(y_test, y_pred)

    sns.heatmap(mat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')

    plt.show()
    
    #save model
    with open(model_path, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
        
    #load model:
    #with open(model_path,'rb') as input:
        #model = pickle.load(input)