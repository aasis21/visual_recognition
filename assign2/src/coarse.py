# -*- coding: utf-8 -*-


import numpy as np
import os
from random import shuffle
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split

dataset_path = './data/'
images_types = os.listdir(dataset_path)

image_data = []
for img_type in images_types:
    img_sub_types = os.listdir(os.path.join(dataset_path,img_type))
    for img_sub_type in img_sub_types:
        images = os.listdir(os.path.join(dataset_path,img_type,img_sub_type))
        for each in images:
            image_data.append((os.path.join(dataset_path,img_type,img_sub_type,each),img_type,img_sub_type))

shuffle(image_data)

images = []
labels = []
sub_labels = []
print(images_types)
ct = 1
for each in image_data:
    if ct % 100 == 0:
        print(ct,len(image_data))
    ct = ct + 1
    img= io.imread(each[0])
    img = resize(img, (224, 224))
    images.append(img)
    labels.append(each[1])
    sub_labels.append(int(each[2]))

classes = np.unique(labels)
print(classes)
nClasses = len(classes)

class_label = {}
cnt = 0

for each in classes:
    class_label[each] = cnt;
    cnt = cnt + 1
    
for i in range(len(labels)):
    labels[i] = class_label[labels[i]]


from keras.utils import to_categorical
labels_one_hot = to_categorical(labels)


train_images, test_images , train_labels, test_labels = train_test_split(images, labels_one_hot , test_size=0.25)
train_images, val_images, train_labels, val_labels =  train_test_split(train_images,train_labels, test_size=0.10)


train_images = np.asarray(train_images, dtype= np.float32)
test_images = np.asarray(test_images, dtype= np.float32)
val_images = np.asarray(val_images, dtype= np.float32)
train_labels = np.asarray(train_labels, dtype= np.float32)
test_labels = np.asarray(test_labels, dtype= np.float32)
val_labels = np.asarray(val_labels, dtype= np.float32)


# Find the shape of input images and create the variable input_shape
nRows,nCols,nDims = train_images.shape[1:]
train_data = train_images.reshape(train_images.shape[0], nRows, nCols, nDims)
test_data = test_images.reshape(test_images.shape[0], nRows, nCols, nDims)
val_data = val_images.reshape(val_images.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)





######################################## model
from keras.applications import VGG16
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224,224, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
    
from keras import models
from keras import layers
from keras import optimizers
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
 
 

model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.Dropout(0.25))


#model.add(Flatten())
#
##model.add(layers.Dense(128))
##model.add(layers.BatchNormalization())
##model.add(layers.Activation("relu"))
#
#
#model.add(Dropout(0.4))
#model.add(Dense(nClasses, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 
#

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(nClasses, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])




from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

datagen_val = ImageDataGenerator(rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        


datagen_val.fit(val_data)
datagen.fit(train_data)


# fits the model on batches with real-time data augmentation:
history  = model.fit_generator(
        datagen.flow(train_data,train_labels, batch_size=10),
        steps_per_epoch=len(train_data) / 10, 
        epochs= 10,
        verbose = 1,
        validation_data = datagen_val.flow(val_data,val_labels, batch_size=10),
        validation_steps = len(val_data)/10
        )
        


a = model.evaluate(test_data, test_labels)


print(a)


print("COURSE FINETUNE")
model.save('course_model_temp.h5')
