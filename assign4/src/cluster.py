# -*- coding: utf-8 -*-

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import skimage.io as io

import numpy as np
import os
import pickle

model = VGG16(weights='imagenet', include_top=False)
model.summary()

f = open('./kmeans.pickle', 'rb')
kmeans = pickle.load(f)
vgg16_feature_list = []


for i, fname in enumerate(os.listdir("./cluster_image")):
    print(i, fname)
    img = image.load_img("./cluster_image/" + fname , target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    img_repr = vgg16_feature_np.flatten()
    predict = kmeans.predict(np.array([img_repr]))
    img = io.imread("./cluster_image/" + fname )
    print(predict)
    io.imshow(img)
    io.show()
    
    

    
    


