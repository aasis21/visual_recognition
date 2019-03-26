# -*- coding: utf-8 -*-



import os, random , pickle
import xml.etree.ElementTree as ET
from skimage import io
from skimage.transform import resize

classes = ('__background__',
           'aeroplane',
           'bottle',
           'chair'
           )

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def background(img, patches):
    print(img.shape)
    for i in range(20):
        x = random.randint(0,img.shape[1] - 50)
        y = random.randint(0,img.shape[0] - 50)
        w = random.randint(45,img.shape[1] - x - 1 )
        h = random.randint(45,img.shape[0] - y - 1)
        patch = (x, y , x+w, y+h)
        candidate = True
        for each in patches:
            iou = get_iou(patch, each)
            if(iou > 0.2):
                candidate = False
            
        p_img = img[y : y + h , x: x + w ]
        if candidate:
            return p_img

    return p_img

def sliding_window(image_s, stepSize, windowSize):
    for y in range(0, image_s[0], stepSize):
        for x in range(0, image_s[1], stepSize):
            yield (x, y, (windowSize[0] ,  windowSize[1]) )
            
def background2(img, patches):
    aspect_ratios = [(400, 400), (224,224), (120, 300) ,(180,100)]
    candidates = []
    for each in aspect_ratios:
        (winH, winW) = each
        for (x, y, window_s) in sliding_window(img.shape, stepSize=32, windowSize=(winH, winW)):
            if window_s[0] != winH or window_s[1] != winW:
                    continue
            candidate = True
            for each in patches:
                iou = get_iou((x, y , x + window_s[1], y + window_s[0]), each)
                if(iou > 0.2):
                    candidate = False
            if candidate:
                candidates.append((x, y , x + window_s[1], y + window_s[0]))
    
    if len(candidates) == 0:
        return background(img, patches)
    
    crd = random.choice(candidates)
    return img[crd[1]: crd[3] , crd[0] : crd[2]]
                
                
                    
                        

resnet_input = [224, 224, 3]

c_dir = os.getcwd()


def build_dataset(typ = "train"):
    print("Building dataset from PASCAL VOC DATASET")
    train_img_addr = c_dir + "/" + "VOC_" + typ + "/JPEGImages"
    train_ann_addr = c_dir + "/" + "VOC_" + typ + "/Annotations"
    train_images = os.listdir(train_img_addr)
    train_id = 0
    train_dict = {}
    g_take_back = 0
    count = [0,0,0,0,0]
    for each in train_images:
        tree =  ET.parse( train_ann_addr + '/' + each[:-3] + 'xml')
        root = tree.getroot()
        objects = []
        take_back = 0
        for obj in root.findall('object'):
            name = obj.find('name').text
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)

            img = io.imread(train_img_addr + '/' + each)
            if name in classes:
                objects.append((xmin, ymin, xmax, ymax))
                take_back =  1
                c_img = img[ ymin:ymax, xmin:xmax]
                r_img = resize(c_img, (resnet_input[0], resnet_input[1]))
                io.imsave(c_dir + "/data/" + typ + "/img" + str(train_id) + ".jpg",r_img)
                train_dict[train_id] = name
                train_id = train_id + 1
                print(train_id)
                if name == "aeroplane":
                    count[0] = count[0] + 1
                if name == "bottle":
                    count[1] = count[1] + 1
                if name == "chair":
                    count[2] = count[2] + 1
                            
        g_take_back = g_take_back + 1
        
        if take_back == 1:
            b_img = background2(img, objects)
            r_img = resize(b_img, (resnet_input[0], resnet_input[1]))
            io.imsave(c_dir + "/data/" + typ + "/img" + str(train_id) + ".jpg",r_img)
            train_dict[train_id] = '__background__'
            train_id = train_id + 1
            count[3] = count[3] + 1
            print("back", train_id)
            
        elif g_take_back % 3 == 0:
            b_img = background2(img, objects)
            r_img = resize(b_img, (resnet_input[0], resnet_input[1]))
            io.imsave(c_dir + "/data/" + typ + "/img" + str(train_id) + ".jpg",r_img)
            train_dict[train_id] = '__background__'
            train_id = train_id + 1
            count[4] = count[4] + 1
            print("back", train_id)
    
    filehandler = open("data/" + typ +".pkl","wb")
    pickle.dump(train_dict,filehandler)
    return count
            
count1 = build_dataset("train")
count2 = build_dataset("test")
print(count1,count2)


