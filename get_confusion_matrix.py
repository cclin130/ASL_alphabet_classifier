# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:26:04 2019

@author: Cheng Lin
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models.vgg import vgg11
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

from CNN_classes2_gpu import ASLLettersDataset, CNN

seed=42
np.random.seed(seed)
torch.manual_seed(seed)

def get_normalized_image(image_path):
    X = Image.open(image_path)
    X = transforms.functional.resize(X, [224,224])
    X_tensor = transforms.functional.to_tensor(X)
    X_tensor = (X_tensor-X_tensor.mean())/X_tensor.std()
    
    return X_tensor

if __name__ == '__main__':
    
    #load model:
    device = torch.device('cpu')
    net = vgg11(num_classes=26)
    net.load_state_dict(torch.load('model_cnn_3000_9.pkl', map_location=device))
    
    with open('mat_3000_9.pkl', 'rb') as input:
        mat = pickle.load(input)
    
    sns.heatmap(mat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')

    plt.show()
    
    img_paths = []
    label = []

    #loop through all folders
    #corpus_path = 'data/asl_alphabet_train'
    print('-------------creating dictionaries for data loaders---------')
    for folder in os.listdir(sys.argv[2]):
        full_path = os.path.join(sys.argv[2], folder)
        for image in os.listdir(full_path):
            print(image)
            temp = os.path.join(full_path, image)

            img_paths.append(temp)
            label.append(folder)

    
    #convert data to np array
    X_paths = np.array(img_paths)

    #convert labels to array and transform into numerical labels
    y_labels = np.array(label)
    le = LabelEncoder()
    le.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',\
         'X', 'Y', 'Z'])

    y = le.transform(y_labels)
    
    #get train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_paths, y,shuffle=True, random_state=seed)
    
    partition = {
                'train':X_train,
                'validation':X_val
            }
    labels = dict(zip(X_paths, y))
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    
    val_set = ASLLettersDataset(partition['validation'], labels)
    val_loader = DataLoader(val_set, **params)
    
        #final results
    print('----------------test model------------------')
    with torch.no_grad():
        test_acc = 0
            
        for local_batch, local_labels in val_loader:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
            test_pred = net(local_batch)
            test_argmax = torch.argmax(test_pred, dim=1)
            test_acc += torch.sum(test_argmax == local_labels).item()
            
            break
            
        print("Test accuracy:", (test_acc/float(len(val_loader.dataset))))
    
    #confusion matrix
    mat = confusion_matrix(local_labels, test_argmax)
    
    #save matrix
    print('-----------------saving matrix-------------------')
    print(mat)
    with open(sys.argv[3], 'wb') as output:
        pickle.dump(mat, output, pickle.HIGHEST_PROTOCOL)