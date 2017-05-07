# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project in Udacity.

In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. I have trained, validated and tested a model using Keras. The model will output a steering angle to an autonomous vehicle.

I used the simulator that Udacity provided where I can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track by the simulator.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This  requires:

* cv2
* tensorflow
* keras
* python3.6

Here I use python3.6 with anaconda3. You can configure your own python environment as you want.

The following resources can be found in this github repository:
* drive.py
* video.py
* clone.py
* model.h5
* video.mp4


The simulator can be downloaded from [simulator](https://github.com/udacity/self-driving-car-sim)

## Data Preprocessing
(1)Record the data from the simulator
	In fact , I found the dataset  given by Udacity is good enough to train the model. However, it needs to record more data if we want to make our model more robust to finish one loop of the track. Here, I record another special scenarios' image to train the model. 
(2)Read the raw data from the CSV file
	We can read the data from the csv file, including the steering data and the path of the image we  need to use. This process can be seen from my code of clone.py
(3)Flipped the raw image
	This is a great strantegy. By this way , more dataset will be generated and the model can learn how to drive in another direction.

## Model architecture
Here, I use the Nvidia Model from the paper[End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) 

[//]: # (Image References)
[image1]: Navidiamodel.png "The architecture of Nvidia Model"

### he architecture of Nvidia Model
This  image shows the architecture of Nvidia Model.

The first layer is used to normalized the input array. The next layers are several convolutional layers to get the feature maps from
the dataset. For example, the first convolutional layer can get 24 feature maps. Then, I flatten the last convolutional layer and add a dropout layer to avoid overfitting. And three fully-collected layers are given and one node will get the steer value in the last stage. By the way ,I the "Relu" unit as the activation in the end-to-end CNN architecture. 

![alt text][image1]

## Model Training

Here, we use the dataset given by Udacity and some dataset recorded by myself to train the model.
some images are as follows:

[//]: # (Image References)
[image2]: pic1.jpg "example of the training image"
[image3]: pic2.jpg "example of the training image"
[image4]: pic3.jpg "example of the training image"
![alt text][image1]


## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

### clone.py
This is the clone python file which is used to read the image and steer angle data to train model and save the model file with the format .h5.  I set the optimizer to 'sgd'  and epochs to 30. You can train your own model by modified the path to read the data in clone.py file.


### model.h5
This is my trained model. This model can make the car run the whole lap of at the autonomous status. You can drive your car using this model or your own model. 

[//]: # (Image References)

[image2]: figure_final.png "The error visualization"

### The error visualization
This  image shows the error in the training and validation procedure.

![alt text][image2]



