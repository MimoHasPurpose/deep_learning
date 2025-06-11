# Setup
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d jangedoo/utkface-new

# Extract
import zipfile
zip = zipfile.ZipFile("/content/utkface-new.zip", 'r')
zip.extractall("/content")
zip.close()

# Imports
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input

# Data prep
folder_path = '/content/utkface_aligned_cropped/UTKFace'

age = []
gender = []
img_path = []
for file in os.listdir(folder_path):
    try:
        age.append(int(file.split('_')[0]))
        gender.append(int(file.split('_')[1]))
        img_path.append(file)
    except:
        continue  # skip bad filenames if any

df = pd.DataFrame({'age': age, 'gender': gender, 'img': img_path})

# Train-test split
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
train_df = df.iloc[:20000]
test_df = df.iloc[20000:]

# Image generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Custom generator
def custom_generator(df, datagen, directory, batch_size=32, target_size=(200,200)):
    while True:
        batch_df = df.sample(n=batch_size)
        gen = datagen.flow_from_dataframe(
            batch_df,
            directory=directory,
            x_col='img',
            y_col=None,
            target_size=target_size,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False
        )
        images = next(gen)
        age = batch_df['age'].values
        gender = batch_df['gender'].values
        yield images, {'age': age, 'gender': gender}

# Model architecture
resnet = ResNet50(include_top=False, input_shape=(200,200,3))
resnet.trainable = False

x = Flatten()(resnet.output)
x = Dense(512, activation='relu')(x)

# Output layers
age_out = Dense(256, activation='relu')(x)
age_out = Dense(1, activation='linear', name='age')(age_out)

gender_out = Dense(256, activation='relu')(x)
gender_out = Dense(1, activation='sigmoid', name='gender')(gender_out)

model = Model(inputs=resnet.input, outputs=[age_out, gender_out])

# Compile
model.compile(optimizer='adam',
              loss={'age': 'mae', 'gender': 'binary_crossentropy'},
              metrics={'age': 'mae', 'gender': 'accuracy'},
              loss_weights={'age': 1, 'gender': 99})

# Train
batch_size = 32
train_generator = custom_generator(train_df, train_datagen, folder_path, batch_size=batch_size)
test_generator = custom_generator(test_df, test_datagen, folder_path, batch_size=batch_size)

steps_per_epoch = len(train_df) // batch_size
validation_steps = len(test_df) // batch_size

model.fit(train_generator,
          steps_per_epoch=steps_per_epoch,
          validation_data=test_generator,
          validation_steps=validation_steps,
          epochs=10)
