# -*- coding: utf-8 -*-

import numpy as np
import os
from random import shuffle
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import keras

dataset_path = './data/aircrafts'
images_types = os.listdir(dataset_path)

image_data = []
for img_type in images_types:
    
    
    images = os.listdir(os.path.join(dataset_path,img_type))
    for each in images:
        image_data.append((os.path.join(dataset_path,img_type,each),img_type))
 

from random import shuffle
shuffle(image_data)

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



# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)
val_labels_one_hot = to_categorical(val_labels)

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()

datagen_val = ImageDataGenerator()

datagen_val.fit(val_data)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_data)



from keras.initializers import glorot_normal
import keras


def outer_product(x):
    """
    calculate outer-products of 2 tensors

        args 
            x
                list of 2 tensors
                , assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
    """
    import keras
    return keras.backend.batch_dot(
                x[0]
                , x[1]
                , axes=[1,1]
            ) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):
    """
    calculate element-wise signed square root

        args
            x
                a tensor
    """
    import keras

    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    """
    calculate L2-norm

        args 
            x
                a tensor
    """
    import keras

    return keras.backend.l2_normalize(x, axis=axis)


def build_model(
     size_heigth=224
    ,size_width=224
    ,no_class=200
    ,no_last_layer_backbone=17
    
    ,name_optimizer="sgd"
    ,rate_learning=1.0
    ,rate_decay_learning=0.0
    ,rate_decay_weight=0.0
    
    ,name_initializer="glorot_normal"
    ,name_activation_logits="softmax"
    ,name_loss="categorical_crossentropy"

    ,flg_debug=True
    ,**kwargs
):
    
    keras.backend.clear_session()
    
    print("-------------------------------")
    print("parameters:")
    for key, val in locals().items():
        if not val == None and not key == "kwargs":
            print("\t", key, "=",  val)
    print("-------------------------------")
    
    ### 
    ### load pre-trained model
    ###
    tensor_input = keras.layers.Input(shape=[size_heigth,size_width,3])
    model_detector = keras.applications.vgg16.VGG16(
                            input_tensor=tensor_input
                            , include_top=False
                            , weights='imagenet'
                        )
    

    ### 
    ### bi-linear pooling
    ###

    # extract features from detector
    x_detector = model_detector.layers[no_last_layer_backbone].output
    shape_detector = model_detector.layers[no_last_layer_backbone].output_shape
    if flg_debug:
         print("shape_detector : {}".format(shape_detector))

    # extract features from extractor , same with detector for symmetry DxD model
    shape_extractor = shape_detector
    x_extractor = x_detector
    if flg_debug:
        print("shape_extractor : {}".format(shape_extractor))
        
    
    # rehape to (minibatch_size, total_pixels, filter_size)
    x_detector = keras.layers.Reshape(
            [
                shape_detector[1] * shape_detector[2] , shape_detector[-1]
            ]
        )(x_detector)
    if flg_debug:
        print("x_detector shape after rehsape ops : {}".format(x_detector.shape))
        
    x_extractor = keras.layers.Reshape(
            [
                shape_extractor[1] * shape_extractor[2] , shape_extractor[-1]
            ]
        )(x_extractor)
    if flg_debug:
        print("x_extractor shape after rehsape ops : {}".format(x_extractor.shape))
        
        
    # outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Lambda(outer_product)(
        [x_detector, x_extractor]
    )
    if flg_debug:
        print("x shape after outer products ops : {}".format(x.shape))
        
        
    # rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
    if flg_debug:
        print("x shape after rehsape ops : {}".format(x.shape))
        
        
    # signed square-root 
    x = keras.layers.Lambda(signed_sqrt)(x)
    if flg_debug:
        print("x shape after signed-square-root ops : {}".format(x.shape))
        
    # L2 normalization
    x = keras.layers.Lambda(L2_norm)(x)
    if flg_debug:
        print("x shape after L2-Normalization ops : {}".format(x.shape))



    ### 
    ### attach FC-Layer
    ###

    if name_initializer != None:
            name_initializer = eval(name_initializer+"()")
            
    x = keras.layers.Dense(
            units=no_class
            ,kernel_regularizer=keras.regularizers.l2(rate_decay_weight)
            ,kernel_initializer=name_initializer
        )(x)
    if flg_debug:
        print("x shape after Dense ops : {}".format(x.shape))
    tensor_prediction = keras.layers.Activation(name_activation_logits)(x)
    if flg_debug:
        print("prediction shape : {}".format(tensor_prediction.shape))

        

    ### 
    ### compile model
    ###
    model_bilinear = keras.models.Model(
                        inputs=[tensor_input]
                        , outputs=[tensor_prediction]
                    )
    
    
    # fix pre-trained weights
    for layer in model_detector.layers:
        layer.trainable = False
          # define optimizers
    opt_adam = keras.optimizers.adam(
                    lr=rate_learning
                    , decay=rate_decay_learning
                )
    opt_rms = keras.optimizers.RMSprop(
                    lr=rate_learning
                    , decay=rate_decay_learning
                )
    opt_sgd = keras.optimizers.SGD(
                    lr=rate_learning
                    , decay=rate_decay_learning
                    , momentum=0.9
                    , nesterov=False
                )
    optimizers ={
        "adam":opt_adam
        ,"rmsprop":opt_rms
        ,"sgd":opt_sgd
    }
    
    model_bilinear.compile(
        loss=name_loss
        , optimizer=optimizers[name_optimizer]
        , metrics=["categorical_accuracy"]
    )
    
    
    
    if flg_debug:
        model_bilinear.summary()
    
    return model_bilinear


model = build_model(
            no_class = nClasses
            ,rate_learning=1.0
            ,rate_decay_weight=1e-8
            ,flg_debug=True
        )
# fits the model on batches with real-time data augmentation:
model.fit_generator(
        datagen.flow(train_data,train_labels_one_hot, batch_size=20),
        steps_per_epoch=len(train_data) / 20, 
        epochs= 20,
        verbose = 1,
        validation_data = datagen_val.flow(val_data,val_labels_one_hot, batch_size=20),
        validation_steps = len(val_data)/20
        )


#
#validation_data = datagen_val.flow(val_data,val_labels_one_hot, batch_size=20),
#        validation_steps = len(val_data)/20
##print("DUMPED")
#model.save('md1.h5')
#
print("START EVAL")
a = model.evaluate(test_data, test_labels_one_hot)
print(a)

y_pred=model.predict(test_data, verbose = 1)

c=0
for i in range(len(y_pred)):
    if(np.argmax(y_pred[i])==np.argmax(test_labels_one_hot[i])):
        c=c+1
print(c*100.0/len(y_pred))


print("DUMPED")
model.save('fine_dogs.h5')

