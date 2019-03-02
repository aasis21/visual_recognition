import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
from itertools import accumulate
import pickle
from numpy import dot
from numpy.linalg import norm

import sys

print(len(sys.argv))
if(len(sys.argv) == 3):
    print("Query image address : " , sys.argv[1], "Save predicted ranks: ", sys.argv[2])
    path = sys.argv[1]
    output = sys.argv[2]
else:
    print("Pass query image address as {python3 predict.py <query_image_addr> <output file>}")
    exit()

dataset_path =  os.path.realpath("train")
images_types = os.listdir(dataset_path)

#out_file = open(output,"w")

image_paths = []
for img_type in images_types:
    print(img_type)
    all_img = os.listdir(os.path.join(dataset_path,img_type))
    for img_n in all_img:
       image_paths.append(str(img_type) + "/" + str(img_n))
        
    
def get_location_and_description(image_paths, train = 0):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.FATAL)

    model = hub.Module('https://tfhub.dev/google/delf/1')

    # The module operates on a single image at a time, so define a placeholder to
    # feed an arbitrary image in.
    image_placeholder = tf.placeholder(
            tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
            'image': image_placeholder,
            'score_threshold': 100.0,
            'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            'max_feature_num': 1000,
            }

    module_outputs = model(module_inputs, as_dict=True)
    
    global_image_paths = []
    if train==1:
        for each in image_paths:
            global_image_paths.append(dataset_path + each)
    else:
        global_image_paths = image_paths
    
    filename_queue = tf.train.string_input_producer(global_image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    full_image_tf = tf.image.decode_jpeg(value, channels=3)
    if train==1:
        crop_image_tf = tf.image.crop_to_bounding_box(full_image_tf, 140, 240, 250, 80 )
    else:
        crop_image_tf = full_image_tf
        
    image_tf = tf.image.convert_image_dtype(crop_image_tf, tf.float32)
    
    with tf.train.MonitoredSession() as sess:
        results_dict = {}  
        for image_path in image_paths:
            image = sess.run(image_tf)
            print('Extracting locations and descriptors from %s' % image_path)
            results_dict[image_path] = sess.run(
                [module_outputs['locations'], module_outputs['descriptors']],
                feed_dict={image_placeholder: image})
            
    return results_dict 

    
with open('model.pickle', 'rb') as f:
    model = pickle.load(f) 

delf = model[0]
    
locations_list= np.concatenate([delf[img][0] for img in image_paths])
descriptors_list = np.concatenate([delf[img][1] for img in image_paths])
descriptor_count_per_image = [ delf[img][0].shape[0] for img in image_paths]
descriptor_count_accumulate = list(accumulate([delf[img][0].shape[0] for img in image_paths]))
d_tree = cKDTree(descriptors_list)


### HELPER 

def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    '''
    Image index to accumulated/aggregated locations/descriptors pair indexes.
    '''
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    accumulated_index_start = None
    accumulated_index_end = None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index-1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
        
    return np.arange(accumulated_index_start,accumulated_index_end)

def get_locations_2_use(image_db_index, k_nearest_indices, accumulated_indexes_boundaries):
    '''
    Get a pair of locations to use, the query image to the database image with given index.
    Return: a tuple of 2 numpy arrays, the locations pair.
    '''
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(locations[i])
                locations_2_use_db.append(locations_list[acc_index])
                break
    return np.array(locations_2_use_query), np.array(locations_2_use_db)
 
   
    
test_dataset_path =  os.path.realpath("test_r")
test_images = os.listdir(test_dataset_path)
    
test_out_path =  os.path.realpath("test_outputs")
for img_testp in test_images:
    print(img_testp)
#    test_img = cv2.imread(c)
#    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    path = os.path.join(test_dataset_path, img_testp)
    
    out = os.path.join(test_dataset_path, img_testp[:-3] + "txt")
    out_file = open(out,"w")

    ## HANDLE QUERY 
    q_delf = get_location_and_description([path])
    locations , descriptors =  q_delf[path]
    
    distances, indices = d_tree.query(descriptors, distance_upper_bound=0.8, k = 12, n_jobs=-1)
    
    # get index of descriptor in db_descriptor_list
    
    unique_indices = np.array(list(set(indices.flatten())))
    unique_indices.sort()
    if unique_indices[-1] == descriptors_list.shape[0]:
        unique_indices = unique_indices[:-1]
    
    # From this descriptor list index, we want image names
    image_indexes =[ np.argmax( [np.array(descriptor_count_accumulate)>index] ) for index in unique_indices ] 
    image_indexes = np.array(list(set(image_indexes)))
    
    print("Candidate Images: ", len(image_indexes))
    
    # Array to keep track of all candidates in database.
    inliers_counts = []
    
    
    for index in image_indexes:
        locations_2_use_query, locations_2_use_db = get_locations_2_use(index, indices, descriptor_count_accumulate)
        # Perform geometric verification using RANSAC.
        _, inliers = ransac(
            (locations_2_use_db, locations_2_use_query), # source and destination coordinates
            AffineTransform,
            min_samples=3,
            residual_threshold=30,
            max_trials=1000)
        
        # If no inlier is found for a database candidate image, we continue on to the next one.
        if inliers is None or len(inliers) == 0:
            continue
        
        # the number of inliers as the score for retrieved images.
        inliers_counts.append({"index": index, "inliers": sum(inliers)})
        print('Found inliers for image {} -> {}'.format(index, sum(inliers)))
       
    
    
    score = {}
    for each in inliers_counts:
        score[image_paths[each["index"]]] = each["inliers"]
        
    score = [(k, score[k]) for k in sorted(score, key=score.get, reverse=True)]
    
    seen_images = []
    counter = 0
    for each in score:
        if str(each[0]) in seen_images:
            continue
        else:
            counter = counter + 1
    #        print(each[0])
            seen_images.append(each[0])
    
    print(len(score), counter)
    
    
    # Handling Query using sift    
    sift = model[1]
    kmeans = sift["Kmeans"]
    all_img_list = sift["all_img_list"] 
    tf_idf_vects = sift["tf_idf_vect"] 
    inverse_count = sift["inverse_count"] 
    
    sift = cv2.xfeatures2d.SIFT_create()
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    hist, bin_edges = np.histogram(labels, bins = range(len(cluster_centers) + 1))
    index = [ 1 if hist[i] < 500 else 1 for i in range(len(hist))]
    num_word = len(cluster_centers)
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB
    
    key_points, description = sift.detectAndCompute(img, None)
    desc_centers_label = kmeans.predict(description)
    hist, bin_edges = np.histogram(desc_centers_label, bins = range(len(cluster_centers) + 1))
    
    tf_idf_vect = [ ( index[i] * hist[i] / len(desc_centers_label)  ) * np.log(3400/(1 + inverse_count[i]))  for i in range(num_word)]
            
    score = {}    
    for img in all_img_list:
        a = tf_idf_vects[img]
        b = tf_idf_vect
        score[img] = dot(a, b)/(norm(a)*norm(b))
    
    score = [(k, score[k]) for k in sorted(score, key=score.get, reverse=True)]
          
    for each in score:
        if str(each[0]) in seen_images:
            continue
        else:
            counter = counter + 1
    #        print(each[0])
            seen_images.append(each[0])
    print(len(score), counter)
    
    for each in image_paths:
        if str(each) in seen_images :
            continue
        else:
            counter = counter + 1
    #        print(each[0])
            seen_images.append(each)
    
    print(len(image_paths), counter)
        
    for each in seen_images:
        out_file.write(each.replace('/','_')  + "\n"  )
    out_file.close()
    
    print(len(seen_images))
    print(len(image_paths))
# -*- coding: utf-8 -*-

