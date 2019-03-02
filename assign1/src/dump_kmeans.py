# -*- coding: utf-8 -*-

import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pickle


pickle_des_list = open("sift_des_list.pickle","rb")
descriptor_list = pickle.load(pickle_des_list)
print(len(descriptor_list))


kmeans =  MiniBatchKMeans(n_clusters=12000, random_state=0,batch_size=50000,verbose=True).fit(descriptor_list)
print("------------------------ Done With Kmeans")
with open('kmeans.pickle', 'wb') as f:
    print("Dumping")
    pickle.dump(kmeans, f, pickle.HIGHEST_PROTOCOL)
   
    
