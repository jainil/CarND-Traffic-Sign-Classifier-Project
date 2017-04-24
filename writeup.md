# **Traffic Sign Recognition**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/image1.png "Visualization"
[image2]: ./output_images/image2.jpg "Traffic Sign 1"
[image3]: ./output_images/image3.jpg "Traffic Sign 2"
[image4]: ./output_images/image4.jpg "Traffic Sign 3"
[image5]: ./output_images/image5.jpg "Traffic Sign 4"
[image6]: ./output_images/image6.png "Traffic Sign 5"

[imagep1]: ./output_images/1.png
[imagep2]: ./output_images/2.png
[imagep3]: ./output_images/3.png
[imagep4]: ./output_images/4.png
[imagep5]: ./output_images/5.png

[imagex1]: ./output_images/x1.png
[imagex2]: ./output_images/x2.png
[imagex3]: ./output_images/x3.png
[imagex4]: ./output_images/x4.png
[imagex5]: ./output_images/x5.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jainil/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of traffic sign images per class in the training dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to keep the images in color, because color can have significant meaning in a traffic sign. As a result I applied only a simple preprocessing step of standardizing the images using this formula: `X = (X - X.mean()) / np.std(X)`.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         |     Description        |
|:---------------------:|:---------------------------------------------:|
| Input | 32x32x3 RGB image |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| Activation| RELU |
| Max pooling | 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5 | 1x1 stride, valid padding, Output = 10x10x16.|
| Activation | RELU |
| Max Pooling | 2x2 stride,  outputs 5x5x6|
| Flatten | output 400|
| Fully Connected | output 120 |
| Activation | RELU |
| Fully Connected | output 84 |
| Activation | RELU |
| Fully Connected | output 43 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer(with learning rate of 0.001), a batch size of 128 and 12 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.982
* test set accuracy of 0.917

I decided to go ahead with using the Lenet architecture. Its original purpose was to classify hand written digits, which is quite similar to a symbol classification of traffic signs. Trying it with minimal modifications, I find that the model performs quite well on the validation set with 98% accuracy and about 92% accuracy on the test set.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]
![alt text][image3]
![alt text][image2]
![alt text][image5]
![alt text][image6]

The first image contains some objects in the background which may add some noise and make classification difficult.

The third image might be difficult to classify because it is a traffic sign with a complicated symbol.

The second, fourth and fifth images are simple and should be easy to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image| Prediction|
|:---------------------:|:-----------------------------:|
| 30 km/h | 30 km/h |
| Right turn | Right turn |
| Pedestrian crossing | General Caution |
| Ahead only | Ahead Only |
| Stop | Stop |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%, which compares favorably to the test model performance of about 90% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the same results with the `top_k` function run on the output of `tf.nn.softmax`. Based on the results of the following plots, it is implied that the model is 100% confident of the output.

![alt text][imagex1]
![alt text][imagex2]
![alt text][imagex3]
![alt text][imagex4]
![alt text][imagex5]
