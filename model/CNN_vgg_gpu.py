# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:17:38 2019

@author: Cheng Lin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:55:34 2019

@author: Cheng Lin

Used pytorch tutorial: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import cv2 as cv
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device('cuda')

class ASLLettersDataset(Dataset):
    def __init__(self, img_paths, labels, img_transform):
        
        self.img_paths = img_paths
        self.labels = labels

        self.colour_jitter = transforms.ColorJitter(brightness=0.6, contrast=0.8, saturation=0.6, hue=0.4)
        self.random_rotate = transforms.RandomRotation(30)
        self.random_crop = transforms.RandomCrop(170)
        self.grayscale = transforms.Grayscale(num_output_channels=1)
        self.resize = transforms.Resize([224,224])
        
        self.img_transform = img_transform
        
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        
        #load data and get label
        X = Image.open(path)
        
        #only augment data if img_transform = True
        if self.img_transform:
            X = self.colour_jitter(X)
            X = self.random_rotate(X)
            X = self.random_crop(X)
        
        #convert PIL image to tensor this normalize
        X = self.resize(X)
        X_tensor = transforms.functional.to_tensor(X)
        X_tensor = (X_tensor-X_tensor.mean())/X_tensor.std()
        
        y = self.labels[path]
        
        return X_tensor, y