# -*- coding: utf-8 -*-

from keras.applications import VGG16   
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

 
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224,224, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers:
    layer.trainable = False
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)

model.add(layers.Permute((3, 1, 2)))
model.add(layers.Reshape((-1, 49 )))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])





import numpy as np
import os
from random import shuffle
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split

dataset_path = './data/aircrafts'
images_types = os.listdir(dataset_path)

image_data = []
for img_type in images_types:
    
    
    images = os.listdir(os.path.join(dataset_path,img_type))
    for each in images:
        image_data.append((os.path.join(dataset_path,img_type,each),img_type))
    
images = []
labels = []


ct = 0
for each in image_data:
    print(ct,len(image_data))
    ct = ct + 1
    img= io.imread(each[0])
    img = resize(img, (224, 224))
    images.append(img)
    labels.append(int(each[1]) - 1)

classes = np.unique(labels)
nClasses = len(classes)

train_images, test_images , train_labels, test_labels = train_test_split(images, labels, test_size=0.20)
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


from keras.utils import to_categorical

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)
val_labels_one_hot = to_categorical(val_labels)

def from_image_to_model_repr(images):
    a = model.predict(images, verbose =1)
    reprs = np.zeros((images.shape[0],49,49))
    i =0
    for each in a:
        print(each.shape)
        A = np.einsum('ij,ik->ijk', each, each)
        A = np.sum(A, axis = 0)
        reprs[i] =  A
        i  = i + 1
    return reprs



train_feature = from_image_to_model_repr(train_data)
test_feature = from_image_to_model_repr(test_data)
val_feature = from_image_to_model_repr(val_data)

train_feature = train_feature.reshape(-1, 49 * 49)
test_feature = test_feature.reshape(-1, 49 * 49)
val_feature = val_feature.reshape(-1, 49 * 49)

fine_model = models.Sequential()
fine_model.add(Dense(64,input_shape=(49*49,) ))
fine_model.add(Dense(32))

fine_model.add(Dense(nClasses  ))
fine_model.summary()

fine_model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


batch_size = 20
epochs = 40
                                                                                
history = fine_model.fit(train_feature, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(val_feature, val_labels_one_hot))


model1.evaluate(test_data, test_labels_one_hot)