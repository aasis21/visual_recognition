# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub

import os
from itertools import accumulate
import pickle



dataset_path = './train/'
images_types = os.listdir(dataset_path)


image_paths = []
for img_type in images_types:
    print(img_type)
    all_img = os.listdir(os.path.join(dataset_path,img_type))
    for img_n in all_img:
       image_paths.append(dataset_path + str(img_type) + "/" + str(img_n))
        
    
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
    
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
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

delf = get_location_and_description(image_paths, train=1)




with open('delf.pickle', 'wb') as f:
    print("Dumping DELF FEATURES")
    pickle.dump(delf, f, pickle.HIGHEST_PROTOCOL)
