# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

import os, random , pickle
import xml.etree.ElementTree as ET
from skimage import io
from skimage.transform import resize

from PIL import Image

resnet_input = [224, 224, 3]

class voc_dataset(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, root_dir, train, transform=None):
        self.train = train
        self.root_dir = root_dir
        self.transform = transform
        if(train is True):
            dict_file = os.path.join(self.root_dir, "train.pkl")
            filehandler = open(dict_file,"rb")
            self.dict = pickle.load(filehandler)
        else:
            dict_file = os.path.join(self.root_dir, "test.pkl")
            filehandler = open(dict_file,"rb")
            self.dict = pickle.load(filehandler)
            
        
    def __len__(self):
        return len(self.dict)
        
    def __getitem__(self, idx):
        if self.train is True:
            img_f = os.path.join(self.root_dir, "train")
        else:
            img_f = os.path.join(self.root_dir, "test")
            
        img_name = os.path.join(img_f, "img" + str(idx) + ".jpg")
#        img = io.imread(img_name)
      
        img = Image.open(img_name)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.dict[idx] == "__background__":
            label = [1,0,0,0]
        if self.dict[idx] == "aeroplane":
            label = [0,1,0,0]
        if self.dict[idx] == "bottle":
            label = [0,0,1,0]
        if self.dict[idx] == "chair":
            label = [0,0,0,1]
            
        return img,  np.array(label)
        
    
batch_size = 10
num_epochs = 20
learning_rate =  0.001
hyp_momentum = 0.9

composed_transform = transforms.Compose([ 
        transforms.Resize(224, 224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
train_dataset = voc_dataset(root_dir='/home/aasis21/sem6/visual_recognition/assign3/data', train=True, transform=composed_transform) # Supply proper root_dir
test_dataset = voc_dataset(root_dir='/home/aasis21/sem6/visual_recognition/assign3/data', train=False, transform=composed_transform) 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

import torchvision.models as models

from torch import nn

resnet18 = models.resnet18(pretrained=True)

# Freeze model weights


resnet18.fc = nn.Linear(resnet18.fc.in_features, 4)

model = resnet18
# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

print(model)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), learning_rate, hyp_momentum)

def train():
    for epoch in range(num_epochs):  
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')

train()

print("sjak")
for data, targets in test_loader:
    log_ps = model(data)
    # Convert to probabilities
    ps = torch.exp(log_ps)
    print(ps.shape())
    
# Find predictions and correct
pred = torch.max(ps, dim=1)
equals = pred == targets
# Calculate accuracy
accuracy = torch.mean(equals)