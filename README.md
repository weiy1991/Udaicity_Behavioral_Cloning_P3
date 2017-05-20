# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project in Udacity.

In this project, I used deep neural networks and convolutional neural networks to clone driving behavior. I have trained, validated and tested a model using Keras. The model will output a steering angle to an autonomous vehicle.

I used the simulator that Udacity provided where I can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then use this model to drive the car autono mously around the track by the simulator.

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

## Data collection maneuver and preprocessing
(1)In fact, the data collection maneuver is very important. First, I collect several loops that car drive on the center of the road, this will tell If I only collect image from the mid-line, the car won't know how to drive when it face the edge of the road. So, I need to collect data that car drive forward the edge of the road and then come back to the center of the road.

(2)Record the data from the simulator
	In fact , I found the dataset  given by Udacity is good enough to train the model. However, it needs to record more data if we want to make our model more robust to finish one loop of the track. Here, I record another special scenarios' image to train the model. 

(3)Read the raw data from the CSV file
	We can read the data from the csv file, including the steering data and the path of the image we  need to use. This process can be seen from my code of clone.py

(4)Flipped the raw image
	This is a great strantegy. By this way , more dataset will be generated and the model can learn how to drive in another direction. For example, the raw data are as follows:

[//]: # (Image References)
[image4]: raw1.jpg  "example of the raw image"
[image5]: raw2.jpg  "example of the raw image"
[image6]: raw3.jpg  "example of the raw image"
![alt text][image4] ![alt text][image5] ![alt text][image6]


Then, I flipped the raw image to augement the dataset, the flipped images are as follows:


[//]: # (Image References)
[image7]: raw1_flipped.jpg "example of the flipped image"
[image8]: raw2_flipped.jpg "example of the flipped image"
[image9]: raw3_flipped.jpg "example of the flipped image"
![alt text][image7] ![alt text][image8] ![alt text][image9]



## Model architecture
Here, I use the Nvidia Model from the paper[End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) 

[//]: # (Image References)
[image1]: Nvidiamodel.png "The architecture of Nvidia Model"

### the architecture of Nvidia Model
This  image shows the architecture of Nvidia Model. But I add a dropout layer to the model to avoid overfitting.

The final model architecture (clone.py lines 44-57) consisted of 5 convolution neural networks with one Lambda layer, one Keras cropping layer, one flatten layer, one dropout layer, 5 dense layers

The Lambda layer is used to normalize the image data and the cropping layer is used to crop the image.

The first convolutional layer can get 24 feature maps with 5x5 filter sizes and a Relu unit(clone.py lines 46).

The second convolutional layer can get 36 feature maps with 5x5 filter sizes and a Relu unit(clone.py lines 47).

The third convolutional layer can get 48 feature maps with 5x5 filter sizes and a Relu unit(clone.py lines 48).

The forth convolutional layer can get 64 feature maps with 3x3 filter sizes and a Relu unit(clone.py lines 49).

The forth convolutional layer can get 64 feature maps with 3x3 filter sizes and a Relu unit(clone.py lines 50).

Next layer is the Kera Flatten layer which is used to flatten the ConvNet.(clone.py lines 51).

Then, the dropout layer is used to avoid overfitting, I set 50% nerual to be obtained.(clone.py lines 52).

The last 5 layers are dense layer, the number of units are 100, 50, 10 ,5,  1.(clone.py lines 53-57).


![alt text][image1]

## Model Training

Here, we use the dataset given by Udacity and some dataset recorded by myself to train the model.
one example image is as follows:

[//]: # (Image References)
[image2]: pic1.jpg "example of the training image"
![alt text][image2]

I split 15% traning dataset to be the validation dataset. meanwhile , I set the epochs to be 30 because I found the loss will be stable after several experiments. Then, I modified the optimizer after got the advice from the mentor, I then use the Adam as the optimizer. I found the optimizer was good from the loss visulization and I could drive the car for one lap of the track after I used the strategy and modified some bug. In a word, this stragety works good in my expriment.

## Problem I met and the way to solve the problem

###Attempts to reduce overfitting in the model
(1)When I first trained my model, I found that the validation loss was high over the time. This implied that the model was overfitting. To combat the overfitting, I modified the model to add a dropout layer so that my model can avoid overfitting, the result was good.

###Attempts to slove some failed cases 
(2) However, there were a few spots where the vehicle fell off the track when I test my model. To improve the driving behavior in these cases, I recorded more data in these cases. On one hand, this would teach the model to learn how to drive through the cases. On another hand, more data could avoid overfitting. After several test, the car could go through the failed cases before.

## Details on model training and characteristics of the dataset

(1)If I used the data only recorded from the center of the road, the car may go out of the road with my model. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the center of the road. These images show what a recovery looks like starting from the road edge. After training the model with these image, the car can go well on the track.

(2)To augment the data sat, I also flipped images and angles thinking that this would augment the dataset and teach the model with two direction. Which means the model can learn more scenarios from the augmentation of images.

(3)After the collection process, I had 14000 images. I then preprocessed this data by flipped the images , then the dataset contains 28000 images.

(4)I finally randomly shuffled the data set and put 15% of the data into a validation set. This will tell me the quality of model.

(5)I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by the loss of validation.


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

[//]: # (Image References)
[image3]: drive.png "example of driving the car by AI"
![alt text][image3]

and you can also see the video I made before on the Youtube [YUAN_AIDriving](https://www.youtube.com/watch?v=ewKzXdj2MTU) 

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





