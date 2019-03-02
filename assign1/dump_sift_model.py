# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import pickle


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



dataset_path = './train/'
images_types = os.listdir(dataset_path)

sift = cv2.xfeatures2d.SIFT_create()

with open('kmeans.pickle', 'rb') as f:
    kmeans = pickle.load(f)

i_count = 3456

cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
hist, bin_edges = np.histogram(labels, bins = range(len(cluster_centers) + 1))
index = [ 1 if hist[i] < 500 else 1 for i in range(len(hist))]
num_word = len(cluster_centers)


image_hist_data = {}
tf_vects = {}
inverse_dict = [[] for i in  range(num_word)]
inverse_count = [0 for i in range(num_word)]

all_img_list = []
for img_type in images_types:
    print(img_type)
   
print("---------------------------------------------------------------------")

for img_type in images_types:
    print(img_type)
    all_img = os.listdir(os.path.join(dataset_path,img_type))
    for img_n in all_img:
        img = cv2.imread(os.path.join(dataset_path, img_type, img_n))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

        x = crop[str(img_type)][0]; w = crop[str(img_type)][1];
        y = crop[str(img_type)][2] - 20 ; h = crop[str(img_type)][3] + 20 ;

        crop_img = img[y:y+h, x:x+w]

        key_points, description = sift.detectAndCompute(crop_img, None)

        if description is None:
            continue
        else:
            all_img_list.append(str(img_type) + "/" +str(img_n))

        desc_centers_label = kmeans.predict(description)
        hist, bin_edges = np.histogram(desc_centers_label, bins = range(num_word + 1))


        image_hist_data[str(img_type) + "/" + str(img_n)] = hist


        tf_vect = [ ( index[i] * hist[i] / len(desc_centers_label) ) for i in range(num_word)]
        tf_vects[str(img_type) + "/" +str(img_n)] = tf_vect


        for clus in desc_centers_label:
            inverse_dict[clus].append(str(img_type) + "/" +str(img_n))
        for i, c in enumerate(hist):
            if c:
                inverse_count[i] = inverse_count[i] + 1

    print("---")
    

tf_idf_vects  = {}
for img_type in images_types:
    print(img_type)
    all_img = os.listdir(os.path.join(dataset_path,img_type))
    for img in all_img_list:
        tf_idf_vects[img] = [ tf_vects[img][i] * np.log(i_count/(1 + inverse_count[i])) for i in range(num_word)]


for i, each in enumerate(inverse_dict):
    inverse_dict[i] = list(set(each))

pickle_for_sift = {}
pickle_for_sift["Kmeans"] = kmeans
pickle_for_sift["all_img_list"] = all_img_list
pickle_for_sift["tf_idf_vect"] = tf_idf_vects
pickle_for_sift["inverse_count"] = inverse_count


with open('sift_model.pickle', 'wb') as f:
    print("Dumping Pickle for Sift")
    pickle.dump(pickle_for_sift, f)
