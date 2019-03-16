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

        self.colour_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.7, saturation=0.3, hue=0.1)
        self.random_rotate = transforms.RandomRotation(20)
        self.random_crop = transforms.RandomCrop(170)
        self.grayscale = transforms.Grayscale(num_output_channels=1)
        
        self.img_transform = img_transform
        
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        
        #load data and get label
        X = Image.open(path)
        
        if self.img_transform:
            X = self.colour_jitter(X)
            X = self.random_rotate(X)
            X = self.random_crop(X)
    
            
            X_mask = Image.new("RGB", (200,200))        
            X_mask.paste(X, (15,15))
            X = X_mask
        
        #convert PIL image to grayscale and then convert to tensor
        X_tensor = transforms.functional.to_tensor(X)
        X_tensor = (X_tensor-X_tensor.mean())/X_tensor.std()
        
        y = self.labels[path]
        
        return X_tensor, y
    
class CNN(nn.Module):        
    def __init__(self):

        super(CNN, self).__init__()
        self.features = self.make_layers(True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(18432, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 26),
            nn.Softmax(dim=0)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    
    def make_layers(self, batch_norm=False):
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        in_channels = 3
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
        