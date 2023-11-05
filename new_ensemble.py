import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from keras.applications import VGG16, ResNet50
from keras.models import Model
from keras.layers import Average, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from glob import glob

IMAGE_SIZE = [224, 224]

train_path = 'Training-Data'
valid_path = 'Testing-Data'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Training-Data',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Testing_Data',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# don't train existing weights
#for layer in vgg16.layers:
#    layer.trainable = False

for layer in vgg16.layers:
    layer.trainable = False

 # useful for getting number of output classes
folders = glob('Training-Data/*')

print("IDHAR ---------------------", vgg16.output)

# our layers - you can add more if you want
#x = Flatten()(vgg16.output)
x = vgg16.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
prediction = Dense(len(folders), activation='softmax')(x)
model1 = Model(inputs=vgg16.inputs, outputs=prediction)
print(model1.summary())


resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


 # useful for getting number of output classes
folders = glob('Training-Data/*')


# our layers - you can add more if you want
x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
prediction = Dense(len(folders), activation='softmax')(x)
model2 = Model(inputs=resnet.inputs, outputs=prediction)
print(model2.summary())


model1.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])
model2.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])
  
#r1 = model1.fit(training_set, validation_data=test_set,epochs=5) 
#r2 = model2.fit(training_set, validation_data=test_set,epochs=5) 

from tensorflow.keras.models import load_model

#model1.save('model_vgg16_ens.h5')
#model2.save('model_res_ens.h5')

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average

model1 = load_model('/mnt/h/capstone/model_vgg16_ens.h5')
model1_1 = Model(inputs=model1.inputs,outputs=model1.outputs,name="vgg")
model2 = load_model('/mnt/h/capstone/model_res_ens.h5')
model2_2 = Model(inputs=model2.inputs,outputs=model2.outputs,name="res")
models = [model1_1,model2_2]
model_input = Input(shape=(224,224,3))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
ensemble_model = Model(inputs=model_input,outputs=ensemble_output,name="ensemble")

print(ensemble_model.summary())
ensemble_model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])

r3 = ensemble_model.fit(training_set,validation_data=test_set,epochs=5)
ensemble_model.save('ensemble_model.h5')