from __future__ import division
import os
import numpy as np
import random
import keras
from scipy import pi
from scipy.misc import imread, imresize
from itertools import islice
import time
import json
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, RemoteMonitor
from keras.layers import Convolution2D, Input
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam
#!pip install tqdm
import tqdm
from keras.models import load_model


def laod_data():

  LIMIT = None

  DATA_FOLDER = 'data/image'
  TRAIN_FILE = 'data/data.txt'
  split=.8

  X = []
  y = []
  y_new = []

  with open(TRAIN_FILE) as fp:
    for line in islice(fp, LIMIT):
      path, angle = line.strip().split()
      full_path = os.path.join(DATA_FOLDER, path)
      X.append(full_path)
      # using angles from -pi to pi to avoid rescaling the atan in the network
      y.append(int(angle)-1)
  y = keras.utils.np_utils.to_categorical(np.array(y))


  images = np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X[2:5000]])
  split_index = int(split * len(X[2:5000]))
  y_new = y[2:5000]
   
  train_X = images[:split_index]
  train_y = y_new[:split_index]
  test_X = images[split_index:]
  test_y = y_new[split_index:]

  return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

def laod_data2(DATA_FOLDER, TRAIN_FILE):

  LIMIT = None
  split=.8

  X = []
  y = []
  images = np.array([])
  y_new = []
  split_index = 0

  with open(TRAIN_FILE) as fp:
    for line in islice(fp, LIMIT):
      path, angle = line.strip().split()
      full_path = os.path.join(DATA_FOLDER, path)
      X.append(full_path)
      # using angles from -pi to pi to avoid rescaling the atan in the network
      y.append(int(angle)-1)
  y = keras.utils.np_utils.to_categorical(np.array(y))


  if(DATA_FOLDER == '/content/drive/Data/data1/image'):
    images = np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X])
    split_index += int(split * len(X))
    y_new = y
  
  elif(DATA_FOLDER == '/content/drive/Data/data2/image'):
    images1 = np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X[2:5000]])
    split_index = int(split * len(X[2:5000]))
    y_new = y[2:5000]

    images2 = np.append( images1, np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X[5000:8000]]) ) 
    split_index += int(split * len(X[5000:8000]))
    y_new.extend( y[5000:8000] )
    
    images = np.append( images2, np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X[8000:10695]]) )
    split_index += int(split * len(X[8000:10695]))
    y_new.extend( y[8000:10695] )
    
  elif(DATA_FOLDER == '/content/drive/Data/data3/image'):
    images1 = np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X])
    split_index += int(split * len(X[2:5000]))
    y_new.extend( y[2:5000] )

    images2 = np.append( np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X[5000:8000]]) )
    split_index += int(split * len(X[5000:8000]))
    y_new.extend( y[5000:8000] )
    
    images3 = np.append( np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X[8000:12000]]) ) 
    split_index += int(split * len(X[8000:12000]))
    y_new.extend( y[8000:12000] )
    
    images = np.append( np.array([np.float32(imresize(imread(im), size=(120, 320))) / 255 for im in X[12000:15800]]) )
    split_index += int(split * len(X[12000:15800]))
    y_new.extend( y[12000:15800] )
  
  train_X = images[:split_index]
  train_y = y_new[:split_index]
  test_X = images[split_index:]
  test_y = y_new[split_index:]

  return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)







def model():
  
  K.set_image_dim_ordering('tf')
    

  inputs = Input(shape=(120, 320, 3))
  conv_1 = Convolution2D(24, 5, 5, activation='relu', name='conv_1', subsample=(2, 2))(inputs)
  conv_2 = Convolution2D(36, 5, 5, activation='relu', name='conv_2', subsample=(2, 2))(conv_1)
  conv_3 = Convolution2D(48, 5, 5, activation='relu', name='conv_3', subsample=(2, 2))(conv_2)
  conv_3 = Dropout(.3)(conv_3)

  conv_4 = Convolution2D(64, 3, 3, activation='relu', name='conv_4', subsample=(1, 1))(conv_3)

  flat = Flatten()(conv_4)

  dense_1 = Dense(1164)(flat)
  dense_1 = Dropout(.3)(flat)
  dense_2 = Dense(100, activation='relu')(dense_1)
  dense_2 = Dropout(.3)(dense_1)
  dense_3 = Dense(50, activation='relu')(dense_2)
  dense_3 = Dropout(.3)(dense_2)
  dense_4 = Dense(10, activation='relu')(dense_3)
  dense_4 = Dropout(.3)(dense_3)

  final = Dense(4, activation="softmax")(dense_4)


  model = Model(input=inputs, output=final)

print('Loading data...')
train_x, train_y, test_x, test_y = laod_data()
print('data1 Done')

print('Loading data...')
train_x, train_y, test_x, test_y = laod_data()
print('data1 Done')
print('Loading model')

model = load_model('model.hdf5')
#	history, nvidia = main(model)









def main(model):
  print('Loading model')

  nvidia = model

  
  nvidia.compile(
    optimizer=Adam(lr=0.0009, clipnorm=.25, beta_1=0.7, beta_2=0.99),
    loss='mse', metrics=['acc'])
  
  checkpointer = ModelCheckpoint(
      filepath="{epoch:02d}-{val_loss:.12f}.hdf5",
      verbose=1,
      save_best_only=True
  )

  lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1, mode=min)

  monitor = RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)

  epochs = 20
  batch_size = 32

  print('Starting training')
  history = nvidia.fit(train_x, train_y,
      validation_data=(test_x, test_y),
      nb_epoch=epochs,
      batch_size=batch_size,
      callbacks=[checkpointer, lr_plateau, monitor]
  )

  print('Done')
  
  hist = pd.DataFrame(history.history)
  plt.figure(figsize=(12,12));
  plt.plot(hist["loss"]);
  plt.plot(hist["val_loss"]);
  plt.plot(hist["acc"]);
  plt.plot(hist["val_acc"]);
  plt.legend()
  plt.show();
  
  return history, nvidia

history, nvidia = main()

print('predict')
nvidia.predict(test_X[355: 370])

test_Y[355:370]

LIMIT = None

DATA_FOLDER = 'data/image'
TRAIN_FILE = 'data/data.txt'

split=.8

X = []
y = []

with open(TRAIN_FILE) as fp:
  for line in islice(fp, LIMIT):
    path, angle = line.strip().split()
    full_path = os.path.join(DATA_FOLDER, path)
    X.append(full_path)
    y.append(int(angle)-1)
y = keras.utils.np_utils.to_categorical(np.array(y))


images = np.array([np.float32(imresize(imread(im), size=(320, 120))) / 255 for im in X[5000:8300]])
split_index = int(split * len(X[5000:8300]))

y_new = y[5000:8300]

train_X = np.array(images[:split_index])
train_Y = np.array(y_new[:split_index])
test_X = np.array(images[split_index:])
test_Y = np.array(y_new[split_index:])

images


print('Loading model')
nvidia = model()

checkpointer = ModelCheckpoint(
    filepath="{epoch:02d}-{val_loss:.12f}.hdf5",
    verbose=1,
    save_best_only=True
)

lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1, mode=min)

monitor = RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)

epochs = 100
batch_size = 64

print('Starting training')
history = nvidia.fit(train_X, train_Y,
    validation_data=(test_X, test_Y),
    nb_epoch=epochs,
    batch_size=batch_size,
    callbacks=[checkpointer, lr_plateau, monitor]
)
print('Done')

hist = pd.DataFrame(history.history)
plt.figure(figsize=(12,12));
plt.plot(hist["loss"]);
plt.plot(hist["val_loss"]);
plt.plot(hist["acc"]);
plt.plot(hist["val_acc"]);
plt.legend()
plt.show();

model = Sequential()

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform', activation='relu'))
model.add(Dropout(0.2))

# Flatten layer
model.add(Flatten())

# Three fully connected layer with dropout on first two
model.add(Dense(100, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(50, init='glorot_uniform', activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, init='glorot_uniform', activation='relu'))

# Output layer, predicts steering angle
model.add(Dense(6, init='glorot_uniform', activation='softmax'))
