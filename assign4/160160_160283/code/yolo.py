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


def draw_and_track(img, boxes):
    global objList, objRect, freeIDS
    newObjID = []
    newObjRect = []
    for box in boxes:
        (x, y, w, h) = box[0]
        class_id = box[1]
        conf = box[2]
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
        color = [int(c) for c in COLORS[class_id]]

        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 1)
        text = "{}: {:.4f}".format(LABELS[class_id], conf)
        cv2.putText(img, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, str(objID), (x + w-20, y+h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 153, 153), 2)

    freeIDS = set([0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14 , 15, 16, 17, 18 ,19 ,20])			
    objList = newObjID
    objRect = newObjRect
    for o in set(objList):
        freeIDS.remove(o)

    return img


def give_yolo_boxes(image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
    	for detection in output:
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
    		if confidence > 0.5:
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))

    			# update our list of bounding box coordinates, confidences,
    			# and class IDs
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			class_ids.append(classID)

    final_output = []
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            final_output.append(([x,y,w,h], class_ids[i], confidences[i] ))
    
    return final_output

weights_path = "./yolo/model/yolov3.weights"
config_path = "./yolo/model/yolov3.cfg"


LABELS = open("./yolo/model/coco.names").read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

freeIDS = set([0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14 , 15, 16, 17, 18 ,19 ,20])			
objList = []		
objRect = []

filename = './Dataset/camera3/JPEGImages/output.mp4'
videogen = skvideo.io.vreader(filename)
count = 0
output_video = []
for frame in videogen:    
    start = time.time()
    
    frame = cv2.resize(frame ,None, fx= 0.75, fy=0.75)

    boxes = give_yolo_boxes(frame)
    
    img = draw_and_track(frame, boxes )

    end = time.time()
    
    print("[INFO : {}] YOLO took {:.6f} seconds".format(count, end - start))
    count = count +  1
    output_video.append(frame)
#
#    cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#    k = cv2.waitKey(15) & 0xff
#    if k == 27:
#        break
#    input()

skvideo.io.vwrite("cam3_out_yolo.mp4", np.array(output_video).astype(np.uint8) )

cv2.destroyAllWindows()
