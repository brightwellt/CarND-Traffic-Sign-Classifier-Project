#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./writeup/barchart1.png "Visualization"
[image2]: ./writeup/grayscaleSign.png "Grayscaling"
[image2b]: ./writeup/grayscaleSign2.png "Grayscaling and Normalization"
[image3]: ./writeup/augmentedSign.png "Augmented Sign"
[image4]: ./writeup/5_ChildrenCrossing.png "Traffic Sign 1"
[image5]: ./writeup/1_roadworks.png "Traffic Sign 2"
[image6]: ./writeup/7_rightofway.png "Traffic Sign 3"
[image7]: ./writeup/4_yield.png "Traffic Sign 4"
[image8]: ./writeup/3_DoNotEnter.png "Traffic Sign 5"
[image9]: ./writeup/featuremap1.png "Feature Map 1"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! This writeup_template.md file should have been supplied as part of a zip file, also containing my Traffic_Sign_Classifier.ipynb project and the output of the project in html format.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4799 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32 * 32 * 3 (RGB data).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed amongst each type of traffic sign. 

![alt text][image1]

The least frequently occurring sign is the 20km/h speed limit sign, whilst the most common is the 50km/h speed limit.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the luminance of an image contains the majority of detail in the image. This way I could reduce the size of my dataset and focus on developing my neural network. For the actual grayscale conversion, I converted the image to YUV and discarded the UV dimensions.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because I wanted to minimize the amount of work required by my neural network to find an optimal solution. Furthermore, a sign will still have a particular type regardless of the brightness or lighting conditions of the original image. Normalizing should help with this.

Here is another image, grayscaled and then normalized
![alt text][image2b]

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because I wanted to improve the robustness of my neural network. Road signs captured by a real camera will be caught at different angles, orientations and positions. They will have variable brightness and contrast. They might be faded, or have graffiti. 
As such I wrote code which randomly rotates, scales, and translates images. Salt and pepper noise can also be added.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following. 
- luminance channel extracted to create grayscale images, 32x32
- each image has several randomly jittered copies made, with rotate, scale, translate and noise functions being randomly applied.
- no allowances are made for class at this time. We could in theory create additional copies of the rarer images.
- all images normalized to zero mean and unit variance


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model was based upon the LeNet architecture seen in earlier lessons.
My final model consisted of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 YUA image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |     									|
| Fully connected		| inputs 400, outputs 120   					|
| RELU					|												|
| Fully connected		| inputs 120, outputs 84    					|
| RELU					|												|
| Fully connected		| inputs 84 , outputs 43    					|
| Softmax				| Cross Entropy									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer. 
- Batch Size was 128. 
- Number of EPOCHS was 10
- mu was 0
- sigma was 0.05
- training rate was 0.002

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

## My final model results were: ##
- training set accuracy of 0.981
- validation set accuracy of 0.940
- test set accuracy of 0.930

*If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?*


My approach was iterative:
- I started with the LeNet solution covered in the course, recreating a version of it.
- I added code to convert my images to grayscale and scale them to mean 0, unit variance
- At this point I was able to run the model and evaluate speed.
- I moved the project onto Amazon Web Services
- I tinkered with the model hyper parameters somewhat, and altering the LeNet structure.
- This was not terribly effective.
- I read the notes more carefully; particularly the paper on "Traffic Sign Recognition with Multi-Scale Convolutional Networks" and elected to augment the dataset.
- This got me to 93%+.
- I continued to tinker with augmenting data and trying out different hyper parameters to identify sweet spots in the model.
- In particular, I didn't want to stray too far from the default LeNet since this is a well known implementation, suitable for tackling this problem.
- I also went back and refined the augmentation functions.

*If a well known architecture was chosen:
- What architecture was chosen?
- Why did you believe it would be relevant to the traffic sign application?
- How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?*

LeNet was chosen because it is specifically designed for character recognition; a problem very similar to that of road sign recognition. It's performance is well documented. 
The final model accuracy values satisfy the rubric for this assignment. Given more time, I would experiment with 
- sharpening the images
- adding more brightness / contrast / noise variations to the jittered data
- balancing the jittered data so that rare signs have more representation.
- changing the layers in the LeNet model; making the first one a 1x1 convolution.  

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, using StreetView around Munich:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

- The first image, Children Crossing, is similar to other signs with human figures on it. It also has strong horizontal lines of asphalt in the background.
- The second image, Roadworks, is skewed slightly.
- The third image, Right Of Way, has white features in the background, disrupting the sign's outline.
- The fourth image, Yield, is upside down and has little in the way of interesting features. 
- The fifth image, Do Not Enter, has a white sticker on it and is slightly faded. The sticker is likely to disrupt the white shape.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
**
Here are the results of the prediction:**

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing		| Children Crossing								| 
| Road work     		| Road work										|
| Right Of Way     		| Right Of Way									|
| Yield					| Yield											|
| Do Not Enter     		| Turn right ahead								|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.2%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22nd code cell of the Ipython notebook.

For the first image, the model is sure that this is a Children Crossing sign (probability of 1.0), and the image does contain this sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Children crossing   							| 
| .00     				| Bicycles crossing								|
| .00					| Priority road									|
| .00	      			| Road work										|
| .00				    | Beware of ice/snow							|


For the second image, the model is sure that this is a Road work sign (probability of 1.0), and the image does contain this sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work										| 
| .00     				| Pedestrians									|
| .00					| Road narrows on the right						|
| .00	      			| Double curve									|
| .00				    | Bicycles crossing								|


For the third image, the model is sure that this is a Right of Way sign (probability of 1.0), and the image does contain this sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right of Way		   							| 
| .00     				| Pedestrians									|
| .00					| Children crossing								|
| .00	      			| Priority road									|
| .00				    | Double curve									|


For the fourth image, the model is sure that this is a Yield sign (probability of 1.0), and the image does contain this sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield											| 
| .00     				| 30km/h Speed Limit							|
| .00					| No vehicles									|
| .00	      			| 50km/h Speed Limit							|
| .00				    | Right of Way									|


For the fifth image, the model is moderately sure that this is a Turn right ahead sign (probability of 0.513). However it is the second probability, No entry, that is correct. The sticker on the sign appears to be causing confusion. Note that I had other sign images available to me, but wanted to throw in a challenging image. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.513         		| Turn right ahead								| 
| 0.267   				| No entry										|
| 0.145					| Turn left ahead								|
| 0.047      			| Ahead only									|
| 0.025				    | Stop											|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is the first Feature Map on one of the images, showing the Children Crossing.
![alt text][image9]
The edges of the sign, and pixels making up the children are being identified.
At this early stage, some of the background noise - the horizontal asphalt - is also being picked up.

