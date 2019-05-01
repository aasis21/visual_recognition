import skvideo.io
import numpy as np
import cv2, time

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

weights_path = "./mask_rcnn/model/frozen_inference_graph.pb"
config_path = "./mask_rcnn/model/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"


LABELS = open("./mask_rcnn/model/object_detection_classes_coco.txt").read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)

freeIDS = set([0,1,2,3,4,5,6,7,8,9,10, 11, 12])			
objList = []		
objRect = []

videogen = skvideo.io.vreader( './Dataset/camera1/JPEGImages/output.mp4')
output_video = []


count = 0
for frame in videogen:
    frame = cv2.resize(frame ,None, fx= 0.75, fy=0.75)
    
    start = time.time()
    
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    img_copy = frame.copy()
    
    newObjID = []
    newObjRect = []
    
    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        if confidence > 0.5:
            # get bounding box via resacling to image size
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            
                		# get mask and rescale and threshold
            mask = masks[i, class_id]
            mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.3)
            
            cRect = (startX , startY , endX , endY )
            objID, maxSim = get_most_similar(cRect, objList, objRect)
            if ((maxSim < 0.60) or (objID < 0)):
                # new object found
                if(len(freeIDS) > 0):
                    objID = next(iter(freeIDS))
                    freeIDS.remove(objID)
                else:
                    print("ERROR: No more free IDs left")
                    objID = 9
                    
            newObjID.append(objID)
            newObjRect.append(cRect)
            cv2.putText(img_copy, str(objID) ,(endX - 20, endY -5  ), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 153, 153), 2)
        
    		# extract the ROI of the image
            roi = img_copy[startY:endY, startX:endX]
    
            int_mask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=int_mask)
            roi = roi[mask]
    
            color = COLORS[class_id]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
            img_copy[startY:endY, startX:endX][mask] = blended
    
    		# draw the bounding box of the instance on the image
            color = [int(c) for c in color]

            cv2.rectangle(img_copy, (startX, startY), (endX, endY), color, 2)
    
            text = "{}: {:.4f}".format(LABELS[class_id], confidence)
            cv2.putText(img_copy, text , (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    freeIDS = set([0,1,2,3,4,5,6,7,8,9,10,11,12])			
    objList = newObjID
    objRect = newObjRect
    for o in set(objList):
        freeIDS.remove(o)      
    
    end= time.time()
    print("[INFO : {}] Mask RCNN took {:.6f} seconds".format(count, end - start))
    count = count +  1
    output_video.append(img_copy)

    
#    cv2.imshow('frame',cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
#    k = cv2.waitKey(15) & 0xff
#    if k == 27:
#        break

skvideo.io.vwrite("cam1_out_mask_rcnn.mp4", np.array(output_video).astype(np.uint8) )

cv2.destroyAllWindows()