#!/usr/bin/env python

import csv
import cv2
import numpy as np

lines = []

##/home/guolindong/MachineLearning/Udacity/simulator-linux/data_record
#/home/guolindong/MachineLearning/Udacity/tensorflow/CarND-Behavioral-Cloning-P3/data/data
#/home/amax/yuanwei/machinelearning/udacity/CarND-Behavioral-Cloning-P3
#/home/amax/yuanwei/machinelearning/udacity/linux_sim

with open('/home/amax/yuanwei/machinelearning/udacity/linux_sim/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '/home/amax/yuanwei/machinelearning/udacity/linux_sim/data/IMG/'+filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

augmented_images = []
augmented_measurements = []
for image,measurement in zip(images,measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# model of nn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dense(84))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss = 'mse',optimizer ='sgd')
history_object = model.fit(X_train, y_train, validation_split = 0.15, shuffle=True, nb_epoch=30)

model.save('model.h5')

#plot the curve
import matplotlib.pyplot as plt
print(history_object.history.keys())
#plot the curve of loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc='upper right')
plt.show()























