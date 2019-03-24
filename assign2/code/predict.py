# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:53:38 2019

@author: max_entropy
"""


import numpy as np
import os
from random import shuffle
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from keras.models import load_model

import pickle


model_aircraft =  load_model('./models/fine_aircrafts_temp0.h5')
pickle_in_a = open("./models/dg_aircrafts_temp0.pickle","rb")
dg_air = pickle.load(pickle_in_a)
print("loaded\n ")


model_bird =  load_model('./models/fine_birds_temp0.h5')
pickle_in_b = open("./models/dg_birds_temp0.pickle","rb")
dg_bird = pickle.load(pickle_in_b)
print("loaded\n ")
        
model_car =  load_model('./models/fine_cars_temp0.h5')
pickle_in_c = open("./models/dg_cars_temp0.pickle","rb")
dg_car = pickle.load(pickle_in_c)
print("loaded\n ")

model_dog =  load_model('./models/fine_dogs_temp0.h5')
pickle_in_d = open("./models/dg_dogs_temp0.pickle","rb")
dg_dog = pickle.load(pickle_in_d)
print("loaded\n ")

model_fl =  load_model('./models/fine_flowers_temp0.h5')
pickle_in_fl = open("./models/dg_flowers_temp0.pickle","rb")
dg_fl = pickle.load(pickle_in_fl)
print("loaded\n ")

 
pred = model.predict(test_data)

class_pred = open("./class.txt", "r")
pred = class_pred.read()
pred = pred.split("\n")
print(len(pred))


output = open("outputj.txt", "w")

for each in pred:
    file_name = each.split(":")[0].strip()
    each_pred= int(each.split(":")[1].strip())
    img= io.imread(file_name)
    img = resize(img, (224, 224))
    test_image = np.asarray(img, dtype= np.float32)
    nRows,nCols,nDims = test_image.shape
    test_data = test_image.reshape(1, nRows, nCols, nDims)

    if each_pred == 0:
        fine_pred = model_aircraft.predict_generator( dg_air.flow(test_data, batch_size=1), steps = 1 )
        what_to_write = str(names[index]) + " " + "aircrafts aircrafts@" + str(np.argmax(fine_pred)) + "\n"
        output.write(what_to_write)
        print(str(index)  + " : " +  what_to_write + "\n")
        
    if each_pred == 1:
        fine_pred = model_bird.predict_generator( dg_bird.flow(test_data, batch_size=1), steps = 1 )   
        what_to_write = str(names[index]) + " " + "birds_ birds_@" + str(np.argmax(fine_pred)) + "\n"
        output.write(what_to_write)       
        print(str(index)  + " : " +  what_to_write + "\n")

    if each_pred == 2:
        fine_pred = model_car.predict_generator( dg_car.flow(test_data, batch_size=1), steps = 1 )
        what_to_write = str(names[index]) + " " + "cars cars@" + str(np.argmax(fine_pred)) + "\n"
        output.write(what_to_write)       
        print(str(index) + " : " +  what_to_write + "\n")

    if each_pred == 3:
        fine_pred = model_dog.predict_generator( dg_dog.flow(test_data, batch_size=1), steps = 1 )
        what_to_write = str(names[index]) + " " + "dogs_ dogs_@" + str(np.argmax(fine_pred)) + "\n"
        output.write(what_to_write)        
        print(str(index) + " : " +  what_to_write + "\n")

    if each_pred == 4:
        fine_pred = model_fl.predict_generator( dg_fl.flow(test_data, batch_size=1), steps = 1 )
        what_to_write = str(names[index]) + " " + "flowers_ flowers_@" + str(np.argmax(fine_pred)) + "\n"
        output.write(what_to_write)
        print(str(index) + " : " +  what_to_write + "\n")       

output.close()
    
