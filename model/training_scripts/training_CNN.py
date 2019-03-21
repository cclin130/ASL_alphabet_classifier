# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:23:58 2019

@author: Cheng Lin
"""

#Cheng Lin
#MAIS 202 Bootcamp final project
#ASL classifier training code (CNN)
#9 Mar 2019

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

from CNN_classes import ASLLettersDataset, CNN

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
      
if __name__ == '__main__':
    
    corpus_path = sys.argv[1]
    max_data_points = int(sys.argv[2])
    max_epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    model_name = sys.argv[5]
    
    img_paths = []
    label = []

    #loop through all folders
    #corpus_path = 'data/asl_alphabet_train'
    print('-------------creating dictionaries for data loaders---------')
    for folder in os.listdir(corpus_path):
        full_path = os.path.join(corpus_path, folder)
        count=0
        for image in os.listdir(full_path):
            count+=1;
            print(image)
            temp = os.path.join(full_path, image)

            img_paths.append(temp)
            label.append(folder)

            if count>=max_data_points: break
    
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
    
    #load data with Dataset + train
    print('-----------------instantiate data loaders-----------------')
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #cudnn.benchmark = True
    
    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    #max_epochs = 20
    input_size = 200*200*3
    output_size = 26
    #learning_rate = 0.001
    
    #Dataloaders
    train_set = ASLLettersDataset(partition['train'], labels)
    train_loader = DataLoader(train_set, **params)
    
    val_set = ASLLettersDataset(partition['validation'], labels)
    val_loader = DataLoader(val_set, **params)

    #instantiate CNN
    net = CNN(output_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Loop over epochs
    print('--------------training with:------------')
    print("epochs=", max_epochs)
    print("learning_rate=", learning_rate)
    print('-'*30)
    
    for epoch in range(max_epochs):
        total_loss=0
        
        # Training
        for local_batch, local_labels in train_loader:
            # Transfer to GPU if available
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    
            # Model computations
            outputs=net(local_batch)
            loss_size = loss_fn(outputs, local_labels)
                        
            optimizer.zero_grad()
            loss_size.backward()
            optimizer.step()
            
            #store loss values
            total_loss += loss_size.item()
            
        average_loss = total_loss / len(train_loader)
            
        # Validation
        with torch.no_grad():
            val_acc=0
            
            for local_batch, local_labels in val_loader:
                # Transfer to GPU if available
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    
                # Model computations
                val_pred = net(local_batch)
                val_argmax = torch.argmax(val_pred, dim=1)
                val_acc += torch.sum(val_argmax == local_labels).item()
                
        print("(epoch, train_loss, val_acc) = ({0}, {1}, {2})".format(epoch, average_loss, val_acc/float(len(val_loader.dataset))))
        
        if average_loss < 0.001: break
    
    #final results
    print('----------------test model------------------')
    with torch.no_grad():
        test_acc = 0
            
        for local_batch, local_labels in val_loader:
            test_pred = net(local_batch)
            test_argmax = torch.argmax(test_pred, dim=1)
            test_acc += torch.sum(test_argmax == local_labels).item()
            
        print("Test accuracy:", (test_acc/float(len(val_loader.dataset))))
    
    #confusion matrix
#    mat = confusion_matrix(y_val, y_pred)
#
#    sns.heatmap(mat, square=True, annot=True, cbar=False)
#    plt.xlabel('predicted value')
#    plt.ylabel('true value')
#
#    plt.show()
    
    #save model
    print('-----------------saving model-------------------')
    with open(model_name, 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
        
    #load model:
    #with open('obj/model_cnn_1000.pkl','rb') as input:
    #    net = pickle.load(input)