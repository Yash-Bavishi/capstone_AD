# Checking Tensorflow
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
"""
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""

import tensorflow as tf
print(tf.__version__)

# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Training-Data'
valid_path = 'Testing-Data'

#vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


 # useful for getting number of output classes
folders = glob('Training-Data/*')

print("IDHAR ---------------------", resnet.output)

# our layers - you can add more if you want
x = Flatten()(resnet.output)
Dropout(0.8)(x)
#prediction = Dense(len(folders), activation='softmax')(x)
prediction = Dense(len(folders), activation='sigmoid')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# view the structure of the model
model.summary()

#cb = EarlyStopping(monitor='val_loss',patience=4)

# tell the model what cost and optimization method to use
# Two -> binary_crossentropy
# multiple -> categorical_crossentropy

model.compile(
  #loss='categorical_crossentropy',
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#train_datagen = ImageDataGenerator(rescale = 1./255,
train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# test_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator()
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Training-Data',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Testing_Data',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
print(training_set)
print(test_set)
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  
)

"""
# save it as a h5 file


from tensorflow.keras.models import load_model
model.save('model_resnet.h5')


from tensorflow.keras.models import load_model
r = load_model("/mnt/h/capstone/model_vgg16.h5")
print(r.summary())

import matplotlib.pyplot as plt

# plot the loss

loss, acc = r.evaluate(test_set, verbose=2)
print(loss,acc)
"""
model.save('model_resnet_224.h5')