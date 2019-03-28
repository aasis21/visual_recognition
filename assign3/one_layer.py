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
from skimage import io
from skimage.transform import resize

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
      
        img = io.imread(img_name)
        
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
num_epochs = 18
learning_rate =  0.001
hyp_momentum = 0.9

composed_transform = transforms.Compose([ 
        transforms.ToPILImage(),
        transforms.Resize(224, 224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
c_dir = os.getcwd()
   
train_dataset = voc_dataset(root_dir= c_dir + '/data', train=True, transform=composed_transform) # Supply proper root_dir
test_dataset = voc_dataset(root_dir= c_dir + '/data', train=False, transform=composed_transform) 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

import torchvision.models as models
import torch
from torch import nn


resnet18 = models.resnet18(pretrained=True)

resnet18.fc = nn.Linear(resnet18.fc.in_features, 4)

model = resnet18

ct = 0 
for child in model.children():
    ct += 1
    if ct < 8:
        for param in child.parameters():
            param.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print("total_params:",total_params)
total_trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad)
print("total_trainable_params:" ,  total_trainable_params)

print(model)

import torch.optim as optim
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(resnet18.parameters(), learning_rate, hyp_momentum)

def train():
    for epoch in range(num_epochs):  
        running_loss = 0.0
        accuracy_sum = 0.0
        
        t_loss = [0,0]
        t_accur = [0,0]
        
        for i, data in enumerate(train_loader, 0):
            
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            
            outputs = outputs.to(device)
            
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            
             #Accuracy
            output = torch.max(outputs, 1)[1]
            label = torch.max(labels, 1)[1]
            correct = (output == label).float().sum()
            accr = correct/output.shape[0] 
            accuracy_sum = accuracy_sum + accr
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f accurcy: %.3f'  %
                      (epoch + 1, i + 1, running_loss / 20, accuracy_sum/ 20 ))
                t_loss[0] = t_loss[0] + running_loss
                t_loss[1] = t_loss[1] +  20
                t_accur[0] = t_accur[0] + accuracy_sum
                t_accur[1] =  t_accur[1] + 20
                running_loss = 0.0
                accuracy_sum = 0.0
        print("EPOCH SUMMARY: loss ", t_loss[0]/t_loss[1], " Accuracy " , t_accur[0]/t_accur[1] )

    print('Finished Training')

train()

torch.save(model.state_dict(), "./model/one_layer_t.pt")




### LOAD TRAINED MODEL
import torchvision.models as models

from torch import nn
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 4)
model = resnet18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load("./model/one_layer_t.pt"))
model.eval()

## Test accuarcy:
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = outputs.to(device)

        _, predicted = torch.max(outputs.data, 1)
        label = torch.max(labels, 1)[1]

        total += labels.size(0)
        correct += (predicted == label).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = outputs.to(device)
        
        _, predicted = torch.max(outputs, 1)
        label = torch.max(labels, 1)[1]

        c = (predicted == label)
        for i in range(c.size(0)):
            class_correct[label[i]] += int(c[i].item())
            class_total[label[i]] += 1



for i in range(4):
    print('Accuracy of %5s : %2d %%' % (
        i , 100 * class_correct[i] / class_total[i]))