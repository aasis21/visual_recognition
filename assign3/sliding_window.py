
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

model.load_state_dict(torch.load("./model/one_layer.pt", map_location='cpu'))
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
    
    img_t = composed_transform(image)
    img = img_t.permute(1, 2, 0).numpy()
    with torch.no_grad():
        inputs = img_t.unsqueeze(0).to(device)
        outputs = model(inputs)
        outputs = outputs.to(device)
        predicted = torch.max(outputs, 1)[1]
        prob = F.softmax(outputs[0], dim=0)
            
    if(predicted[0].item() != 0 and prob[predicted[0].item()] > 0.6):
        boxes.append( (10 , 10 ,(image.shape[1]) - 10 , (image.shape[0]) - 10 ,prob[predicted[0].item()].item(), predicted[0].item() ) )
    
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
#        print("----------------------------------------------------------------")
        to_mult = image.shape[0]/ resized.shape[0]
        aspect_ratios = [(256,256), (96, 156), (156,96), (400, 400)  ]
        for each in aspect_ratios:
            count += 1
            (winH, winW) = each
            for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winH, winW)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                img_t = composed_transform(window)
                img = img_t.permute(1, 2, 0).numpy()
                with torch.no_grad():
                    inputs = img_t.unsqueeze(0).to(device)
                    outputs = model(inputs)
                    outputs = outputs.to(device)
                    predicted = torch.max(outputs, 1)[1]
                    prob = F.softmax(outputs[0], dim=0)
                    
                if(predicted[0].item() != 0 and prob[predicted[0].item()] > 0.6):
                    boxes.append( (x*to_mult ,y*to_mult,(x + window.shape[1]) *to_mult, (y + window.shape[0]) *to_mult,prob[predicted[0].item()].item(), predicted[0].item() ) )
#                    print(x*to_mult ,y*to_mult,(x + window.shape[1]) *to_mult, (y + window.shape[0]) *to_mult,prob[predicted[0].item()].item(), predicted[0].item() )
                    
    print("window Count :", count)

    orig = image.copy()
    img = image.copy()
    boxes = np.array(boxes)
    for (startX, startY, endX, endY, _, _ ) in boxes:
        cv2.rectangle(orig, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0 , 255), 2)

    pick = nms2(boxes, 0.1)
    print("[x] before applying non-maximum, %d bounding boxes" % (boxes.shape[0]) )
    print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)) )    
    for (startX, startY, endX, endY, cfd, lbl) in pick:
        cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)
        cv2.putText(img, str(lbl) ,(int(startX), int(startY)+ 20 ), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 2)
        cv2.putText(img, str(int(cfd*100)) ,(int(endX) - 20, int(endY) - 20 ), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 2)
    for (startX, startY, endX, endY, lbl) in actual_box:
        cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (255, 255, 0), 2)
        cv2.putText(img, str(lbl) ,(int(startX), int(startY) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(150,255,150), 2)
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


ground_dict_a = {}
predict_dict_a = {} 
ground_dict_b = {}
predict_dict_b = {} 
ground_dict_c = {}
predict_dict_c = {} 

#for each in test_image:
#    img_name = "./VOC_test/JPEGImages/0000" + each + ".jpg"
#    give_bounding_box(img_name)

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
#    
#for each in train_images[:500]:
#    img_name = train_img_addr + '/' + each 
#    # ground_truth
#    tree =  ET.parse( train_ann_addr + '/' + each[:-3] + 'xml')
#    root = tree.getroot()
#    objects = [[], [], []]
#    actual_boxes = []
#    for obj in root.findall('object'):
#        name = obj.find('name').text
#        box = obj.find('bndbox')
#        xmin = int(box.find('xmin').text)
#        ymin = int(box.find('ymin').text)
#        xmax = int(box.find('xmax').text)
#        ymax = int(box.find('ymax').text)
#        if name == 'aeroplane':
#            objects[0].append([xmin, ymin, xmax, ymax])
#            actual_boxes.append([xmin, ymin, xmax, ymax, 1])
#        if name == 'bottle':
#            objects[1].append([xmin, ymin, xmax, ymax])
#            actual_boxes.append([xmin, ymin, xmax, ymax, 2])
#        if name == 'chair':
#            objects[2].append([xmin, ymin, xmax, ymax])
#            actual_boxes.append([xmin, ymin, xmax, ymax, 3])
#
#    ground_dict_a[img_name] = objects[0]
#    ground_dict_b[img_name] = objects[1]
#    ground_dict_c[img_name] = objects[2]
#    
#    # prediction
#    objects = [[], [], []]
#    scores = [[], [], []]
#    print("----------------------------------------------------------------")
#    ct += 1
#    print(img_name, ct)
#    print(actual_boxes)
#    pick = give_bounding_box(img_name, actual_boxes)
#    for xmin , ymin, xmax , ymax , score , label  in pick:
#        if label == 1:
#            objects[0].append([xmin, ymin, xmax, ymax])
#            scores[0].append(score)
#        if label == 2:
#            objects[1].append([xmin, ymin, xmax, ymax])
#            scores[1].append(score)
#        if label == 3:
#            objects[2].append([xmin, ymin, xmax, ymax])
#            scores[2].append(score)
#    
#    predict_dict_a[img_name] = { "boxes" : objects[0], "scores" : scores[0] }
#    predict_dict_b[img_name] =  { "boxes" : objects[1], "scores" : scores[1] }
#    predict_dict_c[img_name] =  { "boxes" : objects[2], "scores" : scores[2] }
#    
#import json
#with open('ground_truth_boxes.json', 'w') as fp:
#    json.dump(ground_dict_c, fp)
#with open('predicted_boxes.json', 'w') as fp:
#    json.dump(predict_dict_c, fp)
#    
det_boxes = list()
det_labels = list()
det_scores = list()
true_boxes = list()
true_labels = list()
true_difficulties = list()

from maps import calculate_mAP
with torch.no_grad():
    for each in train_images[:50]:
         img_name = train_img_addr + '/' + each 
         xml_file = train_ann_addr + '/' + each[:-3] + 'xml'
         actual_boxes, act_boxes, act_labels, actual_difficulties = get_ground_truth(xml_file)
         p_boxes, p_labels, p_scores = give_bounding_box(img_name, actual_boxes)
         true_boxes.append(act_boxes)
         true_labels.append(act_labels)
         true_difficulties.append(actual_difficulties)
         det_boxes.append(p_boxes)
         det_labels.append(p_labels)
         det_scores.append(p_scores)
#    
#    det_boxes = [b.to(device) for b in det_boxes]
#    det_labels = [l.to(device) for l in det_labels]
#    det_scores = [b.to(device) for b in det_scores]
#    true_boxes = [b.to(device) for b in true_boxes]    
#    true_labels = [l.to(device) for l in true_labels]
#    true_difficulties = [d.to(device) for d in true_difficulties]     
    
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    print(APs, mAP)
         
         
         