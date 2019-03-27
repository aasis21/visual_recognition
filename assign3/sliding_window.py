
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

import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nms import nms, nms2
import torch.nn.functional as F


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
def pyramid(image, scale, minSize=(60, 60)):
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

model.load_state_dict(torch.load("./model/one_layer_adam.pt", map_location='cpu'))
model.eval()


composed_transform = transforms.Compose([ 
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    

def give_bounding_box(img_name, actual_box):
    image = io.imread(img_name)
    count = 0    
    boxes = []
    
    image_batch = []
    image_shape = []
    
    img_t = composed_transform(image)
    image_batch.append(img_t)
    image_shape.append((10, 10, (image.shape[1]) - 10, (image.shape[0]) - 10))
  
    for resized in pyramid(image, scale=1.8):
        to_mult = image.shape[0]/ resized.shape[0]
#        aspect_ratios = [(400, 400), (224,224),(96, 156), (156,96), (150, 400), (180,100), (120, 66), (150, 50)]
        aspect_ratios = [(200, 200), (96, 256), (156,96)]
        for each in aspect_ratios:
            count += 1
            (winH, winW) = each
            for (x, y, window) in sliding_window(resized, stepSize=50, windowSize=(winH, winW)):
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                img_t = composed_transform(window)
                xmin = x*to_mult
                ymin = y*to_mult
                xmax = (x + window.shape[1]) *to_mult
                ymax = (y + window.shape[0]) *to_mult
                image_batch.append(img_t)
                image_shape.append((xmin, ymin, xmax, ymax))
         
    with torch.no_grad():
        inputs = torch.stack(image_batch).to(device) if len(image_batch) != 0 else torch.tensor(image_batch).to(device)
        outputs = model(inputs)
        outputs = outputs.to(device)
        predicted = torch.max(outputs, 1)[1]
        for i, each in enumerate(image_shape):
            xmin, ymin, xmax, ymax = each
            label = predicted[i].item()
            prob = F.softmax(outputs[i], dim=0)
            score = prob[label].item()
            if label == 0:
                continue
            if score < 0.7 and (label == 2 or label == 3):
#                print("c1")
                continue
#            if label == 2 and (ymax-ymin) < (xmax-xmin):
##                print("c2")
#                continue
            if label == 3 and (1.1 * (ymax-ymin)) < (xmax-xmin):
#                print("c3")
                continue
            boxes.append( (xmin, ymin, xmax, ymax, score, label) )
            

    boxes = np.array(boxes)
    pick = nms2(boxes, 0.1)
#    print("[x] before applying non-maximum, %d bounding boxes" % (boxes.shape[0]) )
#    print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)) )    
    
#
#    img = image.copy()
#    for (startX, startY, endX, endY, cfd, lbl) in pick:
#        cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)
#        cv2.putText(img, str(lbl) ,(int(startX), int(startY)+ 20 ), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 2)
#        cv2.putText(img, str(int(cfd*100)) ,(int(endX) - 20, int(endY) - 20 ), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 2)
#    for (startX, startY, endX, endY, lbl) in actual_box:
#        cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (255, 255, 0), 2)
#        cv2.putText(img, str(lbl) ,(int(startX), int(startY) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(150,255,150), 2)
#        
#    plt.imshow(img)
#    plt.show()

    boxes = list()
    labels = list()
    scores = list()
    for xmin , ymin, xmax , ymax , score , label  in pick:
        boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
        labels.append(float(label))
        scores.append(score)
    
    r_boxes = torch.tensor(boxes) if len(boxes) == 0 else torch.stack(boxes)
    

    return r_boxes , torch.tensor(labels), torch.tensor(scores)
    


test_image = ["53", "55", "58", "59", "84", "85", "90", "97"]
test_i2 = ["178", "108", "127", "128", "144", "151", "157", "172", "181", "185", "195"]
test_i3 = ["412", "418", "441", "447", "467", "473", "487","490"]



c_dir = os.getcwd()
typ = "test"
train_img_addr = c_dir + "/" + "VOC_" + typ + "/JPEGImages"
train_ann_addr = c_dir + "/" + "VOC_" + typ + "/Annotations"
train_images = os.listdir(train_img_addr)

def get_ground_truth(xml_file):
    tree =  ET.parse( xml_file)
    root = tree.getroot()
    boxes = list()
    labels = list()
    difficulties = list()
    actual_boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        
        if name == 'aeroplane':
            boxes.append(torch.Tensor([xmin, ymin, xmax, ymax]))
            labels.append(1.0)
            difficulties.append(0.0)
            actual_boxes.append([xmin, ymin, xmax, ymax, 1])
        if name == 'bottle':
            boxes.append(torch.Tensor([xmin, ymin, xmax, ymax]))
            labels.append(2.0)
            difficulties.append(0.0)
            actual_boxes.append([xmin, ymin, xmax, ymax, 2])

        if name == 'chair':
            boxes.append(torch.Tensor([xmin, ymin, xmax, ymax]))
            labels.append(3.0)
            difficulties.append(0.0)
            actual_boxes.append([xmin, ymin, xmax, ymax, 3])
            
    r_boxes = torch.tensor(boxes) if len(boxes) == 0 else torch.stack(boxes)
    return actual_boxes, r_boxes, torch.tensor(labels), torch.tensor(difficulties)

 
det_boxes = list()
det_labels = list()
det_scores = list()
true_boxes = list()
true_labels = list()
true_difficulties = list()
ct = 0
from maps import calculate_mAP

with torch.no_grad():
    for each in train_images:
         ct += 1
         if ct % 50 == 0:  
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
            print(ct, APs, mAP)
            
         img_name = train_img_addr + '/' + each 
         xml_file = train_ann_addr + '/' + each[:-3] + 'xml'
         actual_boxes, act_boxes, act_labels, actual_difficulties = get_ground_truth(xml_file)
         if len(actual_boxes) == 0:
             continue
#         print(each, ct)
         p_boxes, p_labels, p_scores = give_bounding_box(img_name, actual_boxes)
         true_boxes.append(act_boxes)
         true_labels.append(act_labels)
         true_difficulties.append(actual_difficulties)
         det_boxes.append(p_boxes)
         det_labels.append(p_labels)
         det_scores.append(p_scores)
         
         
# {'chair': 0.2012559026479721, 'bottle': 0.1512237787246704, 'aeroplane': 0.37917467951774597} 0.24388480186462402

         