# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import skimage.io as io
import skvideo.io
import numpy as np
import cv2
import skimage


def getBG(img_file, offset = 1000):
    vid = skvideo.io.vreader(img_file)
    for frame in vid:
         height = frame.shape[0]
         width = frame.shape[1]
         break
     
    bgImage = np.zeros((height,width, 3))
    
    numFrames = 0
    
    for frame in vid:     
        bgImage += frame
        numFrames += 1    
        if (numFrames%50 == 0):
            print(numFrames)
        if (numFrames > offset):
            break
    bgImage = (bgImage/(numFrames*(1.0)))
    bgImage = bgImage.astype(np.uint8)
    return bgImage


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


def get_sim(box1, box2, c_f, l_f):
    img1 = c_f[box1[1]: box1[3], box1[0]:box1[2]]         
    img2 = l_f[box2[1]: box2[3], box2[0]:box2[2]]  
    
    img1 = cv2.resize(img1,(128,128))
    img2 = cv2.resize(img2,(128,128))

#    plt.imshow(img1),plt.show()
#    plt.imshow(img2),plt.show()
    
    # Initiate SIFT detector
    sift = cv2.ORB_create() 
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    if len(kp2) == 0 or len(kp1) == 0:
        return 0
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
        
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
    
#    plt.imshow(img3),plt.show()
    return len(matches)

def get_most_similar(rect, objList, objRect, c_f, l_f ):
	maxSim = 0
	objID = -1
	for i in range(0, len(objList)):
		tmp = get_sim(rect, objRect[i], c_f, l_f)
		if (tmp > maxSim):
			maxSim = tmp
			objID = objList[i]

	return objID, maxSim

    
def draw_and_track(img, c_f, l_f , patches, cnts ):
    global objList, objRect, freeIDS
    BOUNDING_BOX_COLOUR = (255, 0, 0)
    newObjID = []
    newObjRect = []
    for patch in patches :
        (x, y, w, h) = patch[0]			
        cRect = (x, y, x + w, y + h)
        objID, maxSim = get_most_similar(cRect, objList, objRect, c_f, l_f  )
        if ((maxSim < 50) or (objID < 0)):
            # new object found
            if(len(freeIDS) > 0):
                objID = next(iter(freeIDS))
            else:
                print("ERROR: No more free IDs left")
                objID = -1 
        newObjID.append(objID)
        newObjRect.append(cRect)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.putText(img,str(objID),(x + w-20, y+h -5  ), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 153, 153), 1)

    cv2.drawContours(img, cnts, -1, (0,255,0), 1)
    
    for o in objList:
        freeIDS.add(o)
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

filename = './Dataset/camera2/JPEGImages/output.mp4'

#bg = getBG(filename, 1500)
#graybg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)    
#graybg = cv2.resize(graybg, None, fx=0.75, fy=0.75)
#io.imshow(graybg)
#io.show()

freeIDS = set([0,1,2,3,4,5,6,7,8,9,10])			
objList = []		
objRect = []

videogen = skvideo.io.vreader(filename)

last_frame =  cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)

for frame in videogen:
    frame = cv2.resize(frame ,None, fx= 0.75, fy=0.75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = cv2.absdiff(gray, graybg)
#    fgmask[fgmask < 10] = 0
    ret, fgmask = cv2.threshold(fgmask, 40, 255, cv2.THRESH_BINARY)	

	# dilate the thresholded image to fill in holes, then find contours on thresholded image

    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 3)
	
    fgmask_filtered = filter_mask(fgmask)
    
    patches, cnts = detect_vehicles(fgmask_filtered)
    
    img = draw_and_track(frame, frame, last_frame, patches, cnts )
    
    last_frame = gray
#    cv2.imshow('frame',fgmask_filtered)
    cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
