import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import os 
import numpy as np
import cv2

train_directory = "/content/train"
val_directory  = "/content/valid"
image_size = (224,224)

height , width = image_size[0],image_size[1]

def unconvert(width, height, x, y, w, h):
    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    xmin = round(xmin/ width, 2)
    ymin = round(ymin/ height, 2)
    xmax = round(xmax/ width, 2)
    ymax = round(ymax/ height, 2)
    return [xmin, ymin, xmax, ymax]

def readdata(directory):
  images = []
  targets = []
  labels = []
  for filename in os.listdir(os.path.join(directory,"images")):
    train_image_fullpath = os.path.join(directory,"images",filename)
    train_img = cv2.imread(train_image_fullpath)
    train_img = cv2.resize(train_img,(224,224))
    train_img_arr = keras.preprocessing.image.img_to_array(train_img)
    
    filename = filename.replace(".jpg",".txt")
    train_label_full_path = os.path.join(directory,"labels",filename)
    file = open(train_label_full_path, 'r')
    target = list(map(float, file.read().split()))
    i = 0
    if (len(target)%4 ==1 )and (len(target)>=4 ) :
      target_array = []
      images.append(train_img_arr)
      labels.append(target[0])
      target = target[1:]
      x, y, w, h = target[i:i+4]
      target_array.append(unconvert(width, height, x, y, w, h))
      i += 4
      targets.append(target_array)

  return np.array(images),np.array(targets),np.array(labels)
train_images , train_targets ,train_labels = readdata(train_directory)
validation_images , validation_targets ,validation_labels = readdata(val_directory)


num_classes = 7
classes = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

input_shape = (height, width, 3)
input_layer = tf.keras.layers.Input(input_shape)

base_layers = layers.experimental.preprocessing.Rescaling(1./255, name='bl_1')(input_layer)
base_layers = layers.Conv2D(16, 3, padding='same', activation='relu', name='bl_2')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_3')(base_layers)
base_layers = layers.Conv2D(32, 3, padding='same', activation='relu', name='bl_4')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_5')(base_layers)
base_layers = layers.Conv2D(64, 3, padding='same', activation='relu', name='bl_6')(base_layers)
base_layers = layers.MaxPooling2D(name='bl_7')(base_layers)
base_layers = layers.Flatten(name='bl_8')(base_layers)

classifier_branch = layers.Dense(128, activation='relu', name='cl_1')(base_layers)
classifier_branch = layers.Dense(num_classes, name='cl_head')(classifier_branch)  

locator_branch = layers.Dense(256, activation='relu', name='bb_1')(base_layers)
locator_branch = layers.Dense(128, activation='relu', name='bb_2')(locator_branch)
locator_branch = layers.Dense(64, activation='relu', name='bb_3')(locator_branch)
locator_branch = layers.Dense(4, activation='sigmoid', name='bb_head')(locator_branch)
model = tf.keras.Model(input_layer,outputs=[classifier_branch,locator_branch])

losses = {"cl_head":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), "bb_head":tf.keras.losses.MSE}
model.compile(loss=losses, optimizer='Adam', metrics=['accuracy'])
trainTargets = {"cl_head": train_labels,"bb_head": train_targets}
validationTargets = {"cl_head": validation_labels,"bb_head": validation_targets}
training_epochs = 100
history = model.fit(train_images, trainTargets,validation_data=(validation_images, validationTargets),batch_size=4,epochs=training_epochs,shuffle=True,verbose=1)
