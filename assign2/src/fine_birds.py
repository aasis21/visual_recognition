# -*- coding: utf-8 -*-

import numpy as np
import os
from random import shuffle
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split

dataset_path = './data/birds_'
images_types = os.listdir(dataset_path)


image_data = []
for img_type in images_types:
    images = os.listdir(os.path.join(dataset_path,img_type))
    for each in images:
        image_data.append((os.path.join(dataset_path,img_type,each),img_type))
 
shuffle(image_data)

images = []
labels = []

print(images_types)
ct = 1
for each in image_data:
    if ct %100 == 0:
        print(ct,len(image_data))
    ct = ct + 1
    img= io.imread(each[0])
    img = resize(img, (224, 224))
    images.append(img)
    labels.append( str(each[1]))

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


train_images, test_images , train_labels, test_labels = train_test_split(images, labels_one_hot , test_size=0.10)
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

from keras.applications import VGG16, resnet50
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

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


model.add(Conv2D(256, (1, 1), activation="relu"))
model.add(Dropout(0.4))


model.add(Flatten())

model.add(layers.Dense(128))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(Dropout(0.4))

model.add(Dense(nClasses, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

 
# Show a summary of the model. Check the number of trainable parameters
model.summary()




from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=25,                        
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

datagen_val = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True)
        

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)

datagen_val.fit(val_data)
datagen_test.fit(train_data)
datagen.fit(train_data)


# fits the model on batches with real-time data augmentation:
history  = model.fit_generator(
        datagen.flow(train_data,train_labels, batch_size=15),
        steps_per_epoch=len(train_data) / 15, 
        epochs= 20,
        verbose = 1,
        validation_data = datagen_val.flow(val_data,val_labels, batch_size=15),
        validation_steps = len(val_data)/15
        )
        

import pickle
datagen_pickle = open("dg_birds.pickle","wb")
pickle.dump(datagen_test, datagen_pickle)
datagen_pickle.close()

import pickle
pickle_in = open("dg_birds.pickle","rb")
dg = pickle.load(pickle_in)

#a = model.evaluate(test_data, test_labels)

a = model.evaluate_generator( datagen_test.flow(test_data,test_labels, batch_size=10), steps = len(test_data) / 10 )

print(a)

model.save('fine_birds_temp.h5')


from matplotlib import pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
