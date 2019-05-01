# -*- coding: utf-8 -*-

import skvideo.io
import skimage.io
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

def detect_vehicles(fg_mask):
    patches = []
    contours_a = []
    # finding external contours
    im2, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= 25) and ( h >= 40 )
        if not contour_valid:
            continue
        centroid = get_centroid(x, y, w, h)
        patches.append(((x, y, w, h), centroid))
        contours_a.append(contour)
    return patches, contours_a

def get_iou(bb1, bb2):
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

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def get_most_similar(rect, objList, objRect):
	maxSim = 0
	objID = -1
	for i in range(0, len(objList)):
		tmp = get_iou(rect, objRect[i])
		if (tmp > maxSim):
			maxSim = tmp
			objID = objList[i]

	return objID, maxSim

def draw_and_track(img, patches, cnts ):
    global objList, objRect, freeIDS
    BOUNDING_BOX_COLOUR = (255, 0, 0)
    newObjID = []
    newObjRect = []
    for patch in patches :
        (x, y, w, h) = patch[0]			
        cRect = (x, y, x + w, y + h)
        objID, maxSim = get_most_similar(cRect, objList, objRect)
		
        if ((maxSim < 0.60) or (objID < 0)):
            # new object found
            if(len(freeIDS) > 0):
                objID = next(iter(freeIDS))
                freeIDS.remove(objID)
            else:
                print("ERROR: No more free IDs left")
                objID = 15
        newObjID.append(objID)
        newObjRect.append(cRect)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 2)
        cv2.putText(img,str(objID),(x + w-20, y+h -5  ), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 153, 153), 2)

    cv2.drawContours(img, cnts, -1, (0,255,0), 1)
    
    freeIDS = set([0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ,20])			
    objList = newObjID
    objRect = newObjRect
    for o in set(objList):
        freeIDS.remove(o)
        
    return img

def filter_mask(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Fill any small holes
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        erode = cv2.erode(opening, kernel, iterations=1)
        
        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(erode, kernel, iterations=2)
        return dilation


freeIDS = set([0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])			
objList = []		
objRect = []

videogen = skvideo.io.vreader( './Dataset/camera1/JPEGImages/output.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

for frame in videogen:
    frame = cv2.resize(frame ,None, fx= 0.75, fy=0.75)

    fgmask = fgbg.apply(frame)
    
    fgmask[fgmask < 240] = 0
    
    fgmask_filtered = filter_mask(fgmask)
    patches, cnts = detect_vehicles(fgmask_filtered)
    
    img = draw_and_track(frame, patches, cnts)
    
#    image_slic = seg.slic(frame,n_segments=2000)
#    img_final = color.label2rgb(image_slic,frame, kind='avg')
#    cv2.imshow('frame',cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))



    cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#    cv2.imshow('frame',fgmask)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
#    input()
    
    
cv2.destroyAllWindows()