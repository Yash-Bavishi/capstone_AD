import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# Getting the dataset .csv file
data = pd.read_csv("/mnt/h/capstone/Data/DataSet/oasis_cross-sectional.csv")
data = data.copy()
data = data.dropna(subset=["CDR"])
# Classifying 0 is healthy and 1 is AD
data["target"] = np.where(data["CLASS"]=="HC",0,1)

# Droping un-used columns
data = data.drop(columns=['DISC','SES','CDR','eTIV','nWBV','ASF','Delay','CLASS',])

# Splitting dataset using sklearn function
train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# So we need this features to map to keras layers thus we use tf.data
def data_to_tfdata(dataset, shuffle=True, batch_size=32):
    dataset =  dataset.copy()
    labels = dataset.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataset),labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data))
    ds = ds.batch(batch_size)
    return ds
train_ds = data_to_tfdata(data, batch_size=32)
val_ds = data_to_tfdata(data, shuffle=False,batch_size=32)
test_ds = data_to_tfdata(data, shuffle=False,batch_size=32)

print(train_ds)

# To display features
"""
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Age'])
  print('A batch of targets:', label_batch )
"""


example_batch = next(iter(train_ds))[0]
print(next(iter(train_ds))[0])

feature_columns = []

gender = feature_column.categorical_column_with_vocabulary_list('M/F',['M','F'])
gender_class = feature_column.indicator_column(gender)
feature_columns.append(gender_class)

age = feature_column.numeric_column('Age')
feature_columns.append(age)

edu = feature_column.numeric_column('Educ')
edu_bucket = feature_column.bucketized_column(edu, boundaries=[1,2,3,4,5])
print(edu_bucket)

"""
hand = feature_column.categorical_column_with_vocabulary_list('Hand',['L','R'])
hand_class = feature_column.indicator_column(hand)
feature_columns.append(hand_class)

ses = feature_column.numeric_column('SES')
ses_bucket = feature_column.bucketized_column(ses, boundaries=[1,2,3,4])
print(ses_bucket)

mmse = feature_column.numeric_column('MMSE')
mmse_buckets = feature_column.bucketized_column(mmse, boundaries=[0,8,18,25,30])
print(mmse_buckets)
"""

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    # Randomly dropout the neurons to solve the complexity .1 indicates 10%
    layers.Dropout(.1),
    layers.Dense(1)
])

model.compile(optimizer="adam",loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
print("====================READY TO GO BOOM=====================")

model.fit(train_ds,validation_data=val_ds,epochs=100)

print("====================DONE============================")

from tensorflow.keras.models import load_model

model.save('3hl(128,256,128)_model_vgg16_demographic.h5')


loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)


# waited Ensimble Technique 
# 