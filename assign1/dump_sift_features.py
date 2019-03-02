import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import pickle
from sklearn.metrics.pairwise import cosine_similarity


dataset_path = '/home/aasis21/sem6/vr/assign1/train/'
images_types = os.listdir(dataset_path)

sift = cv2.xfeatures2d.SIFT_create()

crop = {}

crop["3m_high_tack_spray_adhesive"] = [260,100,100,175]
crop["aunt_jemima_original_syrup"] = [260,100,75,225]
crop["campbells_chicken_noodle_soup"] = [260,100,200,80]
crop["cheez_it_white_cheddar"] = [240,145,125,165]


crop["cholula_chipotle_hot_sauce"] = [280,70,125,160]
crop["clif_crunch_chocolate_chip"] = [250,125,150,130]
crop["coca_cola_glass_bottle"] = [270,85,80,210]
crop["detergent"] = [230,130,50,250]


crop["expo_marker_red"] = [290,45,175,110]
crop["listerine_green"] = [260,100,50,250]
crop["nice_honey_roasted_almonds"] = [260,100,200,80]
crop["nutrigrain_apple_cinnamon"] = [240,145,150,150]


crop["palmolive_green"] = [280,70,100,170]
crop["pringles_bbq"] = [260,100,90,200]
crop["vo5_extra_body_volumizing_shampoo"] = [260,100,90,200]
crop["vo5_split_ends_anti_breakage_shampoo"] = [260,100,90,200]

    
discriptor_list = np.empty([0,128])
for img_type in images_types:
    print(img_type)
    all_img = os.listdir(os.path.join(dataset_path,img_type))
    count = 0;
    for img in all_img:
        img = cv2.imread(os.path.join(dataset_path, img_type, img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB
        
        x = crop[str(img_type)][0]; w = crop[str(img_type)][1]; 
        y = crop[str(img_type)][2] - 20 ; h = crop[str(img_type)][3] + 20 ;
        
        crop_img = img[y:y+h, x:x+w]
        
        key_points, description = sift.detectAndCompute(crop_img, None)
        if description is not None:
            discriptor_list  = np.append(discriptor_list, description, axis=0)
        
#        sift_img=cv2.drawKeypoints(crop_img,key_points, crop_img)
#        plt.figure(figsize=(5, 5))
#        plt.title('ORB Interest Points')
#        plt.imshow(sift_img); plt.show()
#        
        
        
pickle_des_list = open("sift_des_list.pickle","wb")
pickle.dump(discriptor_list, pickle_des_list)
pickle_des_list.close()        


