
import time
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

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


def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        h = int(image.shape[0] / scale)
        w = int(image.shape[1] / scale)
        image = resize(image, (h,w))
        
        image = image * 255   
        image = image.astype('uint8')         
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[0] or image.shape[1] < minSize[1]:
            break

        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])


# load model

### LOAD TRAINED MODEL
import torchvision.models as models
import torch
from torch import nn
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 4)
model = resnet18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load("./model/one_layer.pt"))
model.eval()


composed_transform = transforms.Compose([ 
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
        

# load the image and define the window width and height
img_name = "/home/max_entropy/Documents/6thsemester/vr/Assignments/visual_recognition/assign3/VOC_test/JPEGImages/000185.jpg"
image = io.imread(img_name)
count = 0
# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    print("----------------------------------------------------------------")
    
    aspect_ratios = [ (128,72), (72,128)]
    for each in aspect_ratios:
        (winH, winW) = each
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winH, winW)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
                                    
            img_t = composed_transform(window)
            img = img_t.permute(1, 2, 0).numpy()
            
            plt.imshow(window)
            plt.show()

            plt.imshow(img)
            plt.show()
            
            with torch.no_grad():
                inputs = img_t.unsqueeze(0).to(device)
                outputs = model(inputs)
                outputs = outputs.to(device)
                predicted = torch.max(outputs, 1)[1]


            print(x,y,predicted)
            count += 1
            time.sleep(0.025)
            
        
print("window Count :", count)