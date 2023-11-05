from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from sklearn.metrics import accuracy_score
from keras.layers import Average
from keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras import Input
from glob import glob
# importing models
vggmodel =  load_model('/mnt/h/capstone/model_vgg16.h5')
resnet = load_model('/mnt/h/capstone/model_resnet_224.h5')

# CHANGING THE LAYER NAME
for layer in vggmodel.layers:
    layer._name = layer.name + str("vgg")

vgg_input = vggmodel.input
res_input = resnet.input


vgg_output = vggmodel.layers[-1].output
res_output = resnet.layers[-1].output

ensemble_output = Average()([vgg_output,res_output])
ensemble_model = Model(inputs=[vgg_input, res_input], outputs=ensemble_output)

ensemble_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

ensemble_model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

res_train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
res_test_datagen = ImageDataGenerator()

# Make sure you provide the same target size as initialied for the image size
vgg_training_set = train_datagen.flow_from_directory('Training-Data',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

vgg_test_set = test_datagen.flow_from_directory('Testing_Data',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
res_training_set = res_train_datagen.flow_from_directory('Training-Data',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

res_test_set = res_test_datagen.flow_from_directory('Testing_Data',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



num_samples = len(vgg_test_set)
print(num_samples)
ensemble_predictions = []
for _ in range(num_samples):
    vgg16_data = vgg_test_set.next()[0]
    resnet50_data = vgg_test_set.next()[0]
    predictions = ensemble_model.predict([vgg16_data, resnet50_data])
    ensemble_predictions.append(predictions)

"""
ensemble_predictions = ensemble_model.predict([vgg_test_set, res_test_set])

r = ensemble_model.fit(
  [vgg_training_set,res_training_set],
  validation_data=[vgg_test_set, res_test_set],
  epochs=20,
  steps_per_epoch=len(vgg_training_set),
  validation_steps=len(vgg_test_set)
)
models = [vggmodel, resnet]
preds = [model.predict(test_set) for model in models]
preds = np.array(preds)
summed = np.sum(preds, axis=0)


ensemble_prediction = np.argmax(summed, axis=1)

vggpred = np.argmax(vggmodel.predict(test_set), axis=-1)
resnetpred = np.argmax(resnet.predict(test_set), axis=-1)
"""
