# -*- coding: utf-8 -*-

import skimage.io as io
import skvideo.io
import numpy as np
import cv2, time
import skimage
from sort.sort import Sort


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
    sort_patches = []
    # finding external contours
    im2, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= 25) and ( h >= 40 )
        if not contour_valid:
            continue
        centroid = get_centroid(x, y, w, h)
        patches.append(((x, y, w, h), centroid))
        sort_patches.append([x,y, x+ w, y+h, 1])
        contours_a.append(contour)
    return patches, contours_a, sort_patches


def draw_and_sort_track(img,track_boxes):
    for box in track_boxes:
        x_s = int(box[0])
        y_s = int(box[1])
        x_e = int(box[2])
        y_e = int(box[3])
        l = int(box[4])
        color = COLORS[ int(l % 100) ]
        color =  (int(color[0]), int(color[1]) , int(color[2]) )        
        cv2.rectangle(img, (x_s, y_s), (x_e , y_e),color , 2 )
        

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



filename = './Dataset/camera3/JPEGImages/output.mp4'



bg = getBG(filename, 1500)
graybg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)    

#graybg = cv2.imread('./Dataset/camera2/JPEGImages/02244.jpg', 0)

graybg = cv2.resize(graybg, None, fx=0.75, fy=0.75)
io.imshow(graybg)
io.show()

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=( 100 , 3), dtype="uint8")

videogen = skvideo.io.vreader(filename)
output_video = []
count = 0
sort_tracker = Sort() 

for frame in videogen:
    frame = cv2.resize(frame ,None, fx= 0.75, fy=0.75)
    start = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = cv2.absdiff(gray, graybg)
#    fgmask[fgmask < 10] = 0
    ret, fgmask = cv2.threshold(fgmask, 40, 255, cv2.THRESH_BINARY)	

	# dilate the thresholded image to fill in holes, then find contours on thresholded image

    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 3)
	
    fgmask_filtered = filter_mask(fgmask)
    
    patches, cnts , sort_detections = detect_vehicles(fgmask_filtered)
    
    track_bbs_ids = sort_tracker.update(np.array(sort_detections))
    
    img = draw_and_sort_track(frame, track_bbs_ids)
    
    end = time.time()
    print("[INFO : {}] CUSTOM BG METHOD took {:.6f} seconds".format(count, end - start))
    count = count +  1
    output_video.append(frame)
    
#    cv2.imshow('frame',fgmask_filtered)
#    cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#    k = cv2.waitKey(15) & 0xff
#    if k == 27:
#        break

skvideo.io.vwrite("cam3_out_custom_bg.mp4", np.array(output_video).astype(np.uint8) )
        
cv2.destroyAllWindows()
