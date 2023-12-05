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

# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import cv2
import sys
import os
#import matplotlib.pyplot as plt

def predictor(mri):
# re-size all the images to this
    IMAGE_SIZE = [224, 224]

    train_path = 'Training-Data'
    valid_path = 'Testing-Data'

    #vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights=None, include_top=False)


    # don't train existing weights
    #for layer in vgg16.layers:
    #    layer.trainable = False

    for layer in vgg16.layers:
        layer.trainable = True

    # useful for getting number of output classes
    folders = glob('/mnt/h/capstone/Training-Data/*')

    # our layers - you can add more if you want
    x = Flatten()(vgg16.output)

    prediction = Dense(len(folders), activation='softmax')(x)
    # create a model object
    model = Model(inputs=vgg16.input, outputs=prediction)
    # view the structure of the model
    try:
        model.load_weights("/mnt/h/capstone/model_vgg16.h5")
    except Exception as e:
        print(f"error {e}", file=sys.stderr)


    #img = image.load_img("/mnt/h/capstone/Training-Data/AD/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc_sag_95.jpg", target_size=(224,224))
    #print(os.listdir("/mnt/h/Capstone-webstack/capstone_backend/public/images"))
    path = "/mnt/h/Capstone-webstack/capstone_backend/public/images/"+os.listdir("/mnt/h/Capstone-webstack/capstone_backend/public/images")[0]
    #path = r"H:\\Capstone-webstack\\capstone_backend\\public\\images\\OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc_sag_95.jpg"
    #print(os.listdir("/mnt/h/Capstone-webstack/capstone_backend/public/images"))
    img = image.load_img(mri,target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    x =  np.expand_dims(x,axis=0)
    #print("IDHAR")
    #img_data = preprocess_input(x)
    #print(model.predict(x))
    a = model.predict(x)
    
    with open("/mnt/h/capstone/output.txt", "w") as file:
        file.write(str(a))
    print(str(a))
    #a=np.argmax(model.predict(x), axis=1)
predictor(sys.argv[1])