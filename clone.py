#!/usr/bin/env python

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []

##/home/guolindong/MachineLearning/Udacity/simulator-linux/data_record
#/home/guolindong/MachineLearning/Udacity/tensorflow/CarND-Behavioral-Cloning-P3/data/data
#/home/amax/yuanwei/machinelearning/udacity/CarND-Behavioral-Cloning-P3
#/home/amax/yuanwei/machinelearning/udacity/linux_sim


############param#########
max_epochs = 200

############### set the plot#############
fig_loss_train=[]
fig_loss_val=[]
fig_train = plt.figure()
fig_x_train = fig_train.add_subplot(111)
Ln_train, = fig_x_train.plot(fig_loss_train, color="blue", linewidth=2.5, linestyle="-", label="train loss")
Ln_val, = fig_x_train.plot(fig_loss_val, color="red", linewidth=2.5, linestyle="-", label="validation loss")
fig_x_train.set_xlim([0,max_epochs])
fig_x_train.set_ylim([0,0.3])
fig_x_train.set_xlabel("epochs")
fig_x_train.set_ylabel("loss value")
plt.ion()
plt.show() 
############end the plot################


def plot_loss_training(Ln_train,Ln_val,fig_loss_train,loss,val_loss,epoch):
	print("plot the new loss, epochs: %d" % (epoch+1))
	fig_loss_train.append(loss)
	fig_loss_val.append(val_loss)
	Ln_train.set_ydata(fig_loss_train)
	Ln_val.set_ydata(fig_loss_val)
	Ln_train.set_xdata(range(len(fig_loss_train)))
	Ln_val.set_xdata(range(len(fig_loss_val)))
	if epoch+1<max_epochs:
		plt.pause(1)
	else:
		plt.pause(0)
		plt.show() 

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
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard, CSVLogger

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss = 'mse',optimizer ='adam')

#history_object = model.fit(X_train, y_train, validation_split = 0.15, shuffle=True, nb_epoch=30)

checkpointer = ModelCheckpoint(filepath="model{epoch:00005d}.h5", verbose=1, save_best_only=False, period = 5)
logger = CSVLogger(filename = "logs/log_clone.csv", separator=',', append=False)
plot_loss_callback = LambdaCallback( on_epoch_end=lambda epoch, logs: \
		plot_loss_training(Ln_train,Ln_val,fig_loss_train,logs['val_loss'],logs['loss'],epoch))
	
history_object = model.fit(X_train, y_train, batch_size = 32, validation_split = 0.15, \
	shuffle=True, nb_epoch=max_epochs, callbacks=[checkpointer,logger,plot_loss_callback])

#model.save('model.h5')

























